"""
astrbot_plugin_uapi - UAPI 百API插件
封装 uapis.cn 100+ 免费 API，支持指令调用和 LLM Tool 调用

指令格式:
  /uapi <api名称> [参数值1] [参数值2] ...
  参数按 API 定义的顺序填入，必填在前、可选在后。

默认将所有可用 API 注册为 LLM Tool（跳过含路径参数的 API），
让大模型自由调用天气、IP、翻译、热榜、二维码、OCR、B站、GitHub 等 90+ 个工具。

示例:
  /uapi misc.weather 北京         查询北京天气
  /uapi network.ipinfo 8.8.8.8   查询IP
  /uapi image.qrcode https://example.com 512
  /uapi translate.text zh 你好世界
"""
import json
import os
import tempfile
from typing import Dict, List, Any

from pydantic import Field
from pydantic.dataclasses import dataclass

from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star
from astrbot.api import html_renderer, logger, AstrBotConfig, FunctionTool
from astrbot.api.message_components import Image, Plain, Record
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.message.message_event_result import MessageChain

from .uapi_client import UAPIClient
from .api_registry import API_DEFINITIONS, API_MAP

PRESENTATION_TEMPLATE = '''<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;background:#edf2f7;font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;color:#172033}
.card{width:760px;box-sizing:border-box;padding:36px 42px;background:linear-gradient(145deg,#fff,#f1f5f9);border-top:8px solid #3b82f6}
.brand{font-size:14px;letter-spacing:1.5px;color:#3b82f6;font-weight:700;margin-bottom:18px}.content{white-space:pre-wrap;font-size:20px;line-height:1.7;word-break:break-word}.content:first-line{font-size:32px;font-weight:800;color:#0f172a}
</style></head><body><main class="card"><div class="brand">UAPI · ASTRBOT</div><div class="content">{{ content }}</div></main></body></html>'''

class UAPIPlugin(Star):
    """UAPI 百API插件 - 封装 100+ 免费 API"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.client = UAPIClient(
            base_url=config.get("base_url", "https://uapis.cn"),
            api_key=config.get("api_key", ""),
            timeout=config.get("request_timeout", 30),
        )
        self._tool_instances = []

        # 注册 LLM Tools
        if config.get("enable_tools", True):
            self._register_llm_tools()

    async def terminate(self):
        """插件卸载时清理资源"""
        await self.client.close()

    # ─────────────────────────────────────────────
    # LLM Tool 注册
    # ─────────────────────────────────────────────

    def _register_llm_tools(self):
        """
        动态注册所有可能的 UAPI 为 LLM Function Tools。

        默认注册 API_MAP 中所有不含路径参数的 API。
        如果配置了 tools_whitelist，则仅注册白名单中指定的 API。
        """
        whitelist_str = self.config.get("tools_whitelist", "").strip()

        registered = 0
        skipped = 0

        if whitelist_str:
            # 白名单模式：只注册指定的 API
            tool_names = [n.strip() for n in whitelist_str.split(",") if n.strip()]
            for name in tool_names:
                api = API_MAP.get(name)
                if not api:
                    logger.warning(f"[UAPI] API '{name}' not found in registry, skipped.")
                    continue
                if self._has_path_params(api):
                    logger.warning(
                        f"[UAPI] API '{name}' has path parameters ({{param}}), "
                        "cannot register as LLM tool, skipped."
                    )
                    skipped += 1
                    continue
                tool_instance = _make_tool_instance(api, self.client)
                if tool_instance:
                    self.context.add_llm_tools(tool_instance)
                    self._tool_instances.append(tool_instance)
                    registered += 1
                    logger.debug(f"[UAPI] Registered LLM tool: {name}")
        else:
            # 全量模式：注册所有不含路径参数的 API
            for name, api in API_MAP.items():
                if self._has_path_params(api):
                    skipped += 1
                    continue
                tool_instance = _make_tool_instance(api, self.client)
                if tool_instance:
                    self.context.add_llm_tools(tool_instance)
                    self._tool_instances.append(tool_instance)
                    registered += 1
                    logger.debug(f"[UAPI] Registered LLM tool: {name}")

        logger.info(
            f"[UAPI] Registered {registered} LLM tools"
            + (f" (skipped {skipped} with path params)" if skipped else "")
        )

    @staticmethod
    def _has_path_params(api: dict) -> bool:
        """检查 API 是否包含路径参数（{param}），这类 API 无法作为 LLM Tool 正确调用"""
        for p in api.get("parameters", []):
            if p.get("in") == "path":
                return True
        return False

    # ─────────────────────────────────────────────
    # 指令处理
    # ─────────────────────────────────────────────

    @filter.command("uapi")
    async def uapi_command(self, event: AstrMessageEvent):
        """
        UAPI 百API调用指令。

        用法:
          /uapi <api名称> <参数值1> <参数值2> ...
          参数按 API 定义顺序填入，必填参数在前，可选参数在后。

          /uapi help <API名称>      查看 API 详情及参数顺序
          /uapi list [页码]         列出所有可用 API
          /uapi search <关键词>     搜索 API
        """
        message_str = event.message_str.strip()
        args_str = message_str[len("/uapi"):].strip()

        if not args_str or args_str == "help":
            yield event.plain_result(self._get_help_text())
            return

        parts = self._parse_args(args_str)

        if not parts:
            yield event.plain_result(self._get_help_text())
            return

        sub_cmd = parts[0].lower()

        # /uapi list [页码]
        if sub_cmd == "list":
            page = 1
            if len(parts) >= 2:
                try:
                    page = int(parts[1])
                except ValueError:
                    pass
            yield event.plain_result(self._list_apis(page))
            return

        # /uapi search <关键词>
        if sub_cmd == "search":
            keyword = " ".join(parts[1:]) if len(parts) > 1 else ""
            yield event.plain_result(self._search_apis(keyword))
            return

        # /uapi help <API名称> - 查看 API 详情
        if sub_cmd == "help":
            if len(parts) < 2:
                yield event.plain_result("用法: /uapi help <API名称>\n例如: /uapi help misc.weather")
                return
            help_api_name = parts[1]
            help_api = API_MAP.get(help_api_name)
            if not help_api:
                yield event.plain_result(f"未找到 API '{help_api_name}'。\n使用 /uapi list 查看所有可用 API。")
                return
            yield event.plain_result(self._format_api_info(help_api))
            return

        # /uapi <api_name> [参数...]
        api_name = parts[0]
        api = API_MAP.get(api_name)

        if not api:
            suggestions = self._fuzzy_find(api_name)
            if suggestions:
                yield event.plain_result(
                    f"未找到 API '{api_name}'。\n\n你可能想找：\n" +
                    "\n".join(f"  \u2022 {s}" for s in suggestions[:10]) +
                    f"\n\n使用 /uapi search {api_name} 搜索更多。"
                )
            else:
                yield event.plain_result(
                    f"未找到 API '{api_name}'。\n使用 /uapi list 查看所有可用 API。"
                )
            return

        # 没有参数时
        if len(parts) == 1:
            ordered_params = self._get_ordered_params(api)
            has_required = any(p.get("required") for p in ordered_params)

            if has_required:
                yield event.plain_result(
                    f"{api['short_name']} 需要参数才能调用。\n"
                    f"使用 /uapi help {api['short_name']} 查看参数说明与顺序。"
                )
                return
            else:
                # 无必填参数，直接调用
                result = await self._call_api(event, api, [])
                yield result
                return

        # 有参数时：位置参数模式
        result = await self._call_api(event, api, parts[1:])
        yield result

    # ─────────────────────────────────────────────
    # 参数解析
    # ─────────────────────────────────────────────

    def _parse_args(self, s: str) -> List[str]:
        """解析参数字符串，支持引号包裹的值"""
        parts = []
        current = ""
        in_quote = False
        quote_char = None

        for ch in s:
            if in_quote:
                if ch == quote_char:
                    in_quote = False
                else:
                    current += ch
            elif ch in ('"', "'"):
                in_quote = True
                quote_char = ch
            elif ch == " ":
                if current:
                    parts.append(current)
                    current = ""
            else:
                current += ch

        if current:
            parts.append(current)

        return parts

    def _get_ordered_params(self, api: dict) -> List[dict]:
        """获取 API 参数顺序列表: 必填Query→可选Query→必填Body→可选Body"""
        ordered = []

        for p in api.get("parameters", []):
            ordered.append({**p, "_source": "query"})

        for p in api.get("body_params", []):
            ordered.append({**p, "_source": "body"})

        # 按必填在前、可选在后排序（同类别内保持原始顺序）
        ordered.sort(key=lambda p: 0 if p.get("required") else 1)
        return ordered

    def _convert_param_value(self, val: str, ptype: str, param_info: dict) -> Any:
        """将字符串参数值转换为目标类型"""
        if ptype == "boolean":
            lower = val.lower()
            if lower in ("true", "1", "yes", "是"):
                return True
            elif lower in ("false", "0", "no", "否"):
                return False
            # 有 enum 限定但值不在其中时原样返回
            enum_vals = param_info.get("enum")
            if enum_vals and val not in enum_vals:
                return val
            return lower == "true"

        if ptype in ("integer", "int"):
            try:
                return int(val)
            except ValueError:
                return val

        if ptype in ("number", "float"):
            try:
                return float(val)
            except ValueError:
                return val

        return val

    # ─────────────────────────────────────────────
    # API 调用
    # ─────────────────────────────────────────────

    async def _call_api(
        self, event: AstrMessageEvent, api: dict, arg_values: List[str]
    ) -> MessageEventResult:
        """
        按位置参数顺序调用 API。
        arg_values 为空列表时表示调用无参数 API。
        """
        ordered_params = self._get_ordered_params(api)

        query_args = {}
        body_args = {}

        # 检查参数数量
        if len(arg_values) > len(ordered_params):
            return event.plain_result(
                f"参数过多！{api['short_name']} 最多接受 {len(ordered_params)} 个参数。\n\n"
                f"使用 /uapi help {api['short_name']} 查看参数说明。"
            )

        for i, val in enumerate(arg_values):
            param_info = ordered_params[i]
            pname = param_info["name"]
            ptype = param_info.get("type", "string")
            source = param_info.get("_source", "query")
            converted = self._convert_param_value(val, ptype, param_info)

            if source == "query":
                query_args[pname] = converted
            else:
                body_args[pname] = converted

        # 检查缺失的必填参数
        missing = []
        for i, p in enumerate(ordered_params):
            if p.get("required") and i >= len(arg_values):
                missing.append(p["name"])

        if missing:
            usage = self._format_positional_usage(api)
            return event.plain_result(
                f"缺少必填参数: {', '.join(missing)}\n\n"
                f"正确用法:\n{usage}"
            )

        method = api["method"]
        path = api["path"]
        logger.info(f"[UAPI] Calling {method} {path} query={query_args} body={body_args}")

        try:
            result = await self.client.call(
                path=path,
                method=method,
                params=query_args if query_args else None,
                json_body=body_args if body_args and not api.get("form_data") else None,
                form_data=body_args if body_args and api.get("form_data") else None,
            )
        except Exception as e:
            logger.error(f"[UAPI] Error calling {path}: {e}")
            return event.plain_result(f"API 调用异常: {str(e)}")

        if not result.get("success"):
            error_data = result.get("data")
            error_msg = result.get("error")
            if not error_msg and isinstance(error_data, dict):
                error_msg = error_data.get("message") or error_data.get("code")
            if not error_msg:
                error_msg = str(error_data or "未知错误")
            return event.plain_result(f"API 调用失败: {error_msg}")

        if result.get("is_binary"):
            return await self._send_binary_result(event, result)

        # JSON/text 响应
        data = result.get("data", {})
        presentation = api["short_name"]
        is_hotboard = presentation == "misc.hotboard"
        is_weather = presentation == "misc.weather"
        is_search = presentation == "search.aggregate"
        if isinstance(data, (dict, list)):
            if is_hotboard and isinstance(data, dict):
                items = data.get("list", data.get("results", []))
                if isinstance(items, list):
                    platform = data.get("type", query_args.get("type", ""))
                    update_time = data.get("update_time", "")
                    formatted = f"# {platform} 热榜"
                    if update_time:
                        formatted += f"\n\n更新时间：{update_time}"
                    for position, item in enumerate(items, start=1):
                        if not isinstance(item, dict):
                            continue
                        title = str(item.get("title", "未命名条目")).replace("\n", " ")
                        rank = item.get("index", position)
                        hot_value = item.get("hot_value")
                        url = item.get("url")
                        formatted += f"\n\n## {rank}. {title}"
                        if hot_value:
                            formatted += f"\n热度：{hot_value}"
                        if url:
                            formatted += f"\n链接：{url}"
                else:
                    formatted = json.dumps(data, ensure_ascii=False, indent=2)
            elif is_weather and isinstance(data, dict):
                location = " ".join(
                    str(data[key])
                    for key in ("province", "city", "district")
                    if data.get(key)
                )
                formatted = f"# {location or '天气'}\n\n"
                formatted += f"## {data.get('weather', '未知')}  {data.get('temperature', '--')}°C"
                for label, key, suffix in (
                    ("体感温度", "feels_like", "°C"),
                    ("湿度", "humidity", "%"),
                    ("风况", "wind_direction", ""),
                    ("风力", "wind_scale", ""),
                    ("能见度", "visibility", " km"),
                    ("紫外线", "uv_index", ""),
                    ("空气质量", "aqi", ""),
                    ("更新时间", "update_time", ""),
                ):
                    if data.get(key) is not None:
                        formatted += f"\n\n**{label}**：{data[key]}{suffix}"
            elif is_search and isinstance(data, dict):
                formatted = f"# 搜索：{data.get('query', '')}"
                if total := data.get("total_results"):
                    formatted += f"\n\n共找到 {total} 条结果"
                for position, item in enumerate(data.get("results", []), start=1):
                    if not isinstance(item, dict):
                        continue
                    formatted += f"\n\n## {item.get('position', position)}. {item.get('title', '未命名结果')}"
                    if domain := item.get("domain"):
                        formatted += f"\n来源：{domain}"
                    if snippet := item.get("snippet"):
                        formatted += f"\n{snippet}"
                    if url := item.get("url"):
                        formatted += f"\n链接：{url}"
            else:
                formatted = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            formatted = str(data)

        if is_hotboard or is_weather or is_search:
            try:
                image_path = await html_renderer.render_custom_template(
                    PRESENTATION_TEMPLATE,
                    {"content": formatted.replace("# ", "").replace("## ", "").replace("**", "")},
                    return_url=False,
                    options={"type": "png", "quality": 90},
                )
                event.track_temporary_local_file(image_path)
                return event.image_result(image_path)
            except Exception as e:
                logger.warning(f"[UAPI] Failed to render long result: {e}")
                formatted = formatted[:1950] + "\n\n... (内容过长已截断)"

        if len(formatted) > 2000:
            return event.plain_result(
                f"{api['summary']}返回内容较长，当前尚未适配聊天展示。"
                "请查看插件日志或等待该接口的专用展示器。"
            )

        return event.plain_result(formatted)

    async def _send_binary_result(
        self, event: AstrMessageEvent, result: dict
    ) -> MessageEventResult:
        """处理二进制响应（图片/音频），正确发送到聊天"""
        data = result["data"]
        content_type = result.get("content_type", "application/octet-stream")

        temp_dir = os.path.join(os.getcwd(), "data", "temp")
        os.makedirs(temp_dir, exist_ok=True)

        is_audio = "audio/" in content_type
        is_image = "image/" in content_type

        if is_audio:
            ext_map = {
                "audio/mpeg": ".mp3",
                "audio/wav": ".wav",
                "audio/ogg": ".ogg",
                "audio/flac": ".flac",
                "audio/aac": ".aac",
                "audio/mp4": ".m4a",
                "audio/webm": ".webm",
            }
            ext = ".mp3"
            for ct, e in ext_map.items():
                if ct in content_type:
                    ext = e
                    break

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=ext, dir=temp_dir
                ) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name

                # 将临时文件生命周期交给框架管理：框架会在 respond 阶段读取文件
                # 发送完毕后，于 PipelineScheduler.execute() 的 finally 中统一调用
                # cleanup_temporary_local_files() 清理。
                # 注意：不能在 finally 中提前 os.unlink()，否则 respond 阶段读取
                # 文件时会因文件已被删除而发送失败（FileNotFoundError）。
                event.track_temporary_local_file(tmp_path)

                # 使用 Record 组件代替 File，确保语音能内联播放
                chain = [
                    Plain("🔊 "),
                    Record.fromFileSystem(tmp_path),
                ]
                return event.chain_result(chain)
            except Exception as e:
                logger.error(f"[UAPI] Failed to save audio: {e}")
                return event.plain_result(f"音频获取成功，但发送失败: {str(e)}")

        if is_image:
            ext_map = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/gif": ".gif",
                "image/webp": ".webp",
                "image/svg+xml": ".svg",
                "image/bmp": ".bmp",
            }
            ext = ".png"
            for ct, e in ext_map.items():
                if ct in content_type:
                    ext = e
                    break

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=ext, dir=temp_dir
                ) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name

                # 与音频分支相同：登记临时文件，交由框架在 pipeline 结束后统一清理，
                # 避免 respond 阶段发送前文件已被删除。
                event.track_temporary_local_file(tmp_path)

                return event.image_result(tmp_path)
            except Exception as e:
                logger.error(f"[UAPI] Failed to save image: {e}")
                return event.plain_result(f"图片获取成功，但发送失败: {str(e)}")

        return event.plain_result(
            f"API 调用成功（二进制数据，{len(data)} bytes，类型: {content_type}）"
        )

    # ─────────────────────────────────────────────
    # 帮助信息
    # ─────────────────────────────────────────────

    def _get_help_text(self) -> str:
        return (
            "UAPI 百API插件 使用指南\n\n"
            "/uapi <API名称> [参数值...]   调用 API，参数按顺序填入\n"
            "/uapi help <API名称>          查看 API 参数详情及顺序\n"
            "/uapi list [页码]              列出所有可用 API\n"
            "/uapi search <关键词>          搜索 API\n\n"
            "示例:\n"
            "  /uapi misc.weather 北京\n"
            "  /uapi network.ipinfo 8.8.8.8\n"
            "  /uapi image.qrcode https://example.com 512\n"
            "  /uapi translate.text zh 你好世界\n"
            "  /uapi network.dns github.com A\n"
            "  /uapi misc.phoneinfo 13800138000\n"
            "  /uapi misc.hotboard weibo\n"
            "  /uapi saying.random daily\n"
            "  /uapi dictionary.audio hello\n\n"
            "提示: 使用 /uapi help <名称> 查看每个 API 的参数顺序"
        )

    def _list_apis(self, page: int = 1, per_page: int = 20) -> str:
        """分页列出所有 API"""
        total = len(API_DEFINITIONS)
        total_pages = (total + per_page - 1) // per_page
        page = max(1, min(page, total_pages))
        start = (page - 1) * per_page
        end = start + per_page

        lines = [f"UAPI API 列表 (第 {page}/{total_pages} 页，共 {total} 个)\n"]
        for api in API_DEFINITIONS[start:end]:
            name = api["short_name"]
            summary = api.get("summary", "")
            lines.append(f"  [{api['method']:4s}] {name:40s} {summary}")

        lines.append(f"\n输入 /uapi help <名称> 查看参数详情。输入 /uapi list {page+1} 翻页。")
        return "\n".join(lines)

    def _search_apis(self, keyword: str, limit: int = 20) -> str:
        """搜索 API"""
        if not keyword:
            return "请输入搜索关键词。例如: /uapi search 天气"

        keyword_lower = keyword.lower()
        results = []
        for api in API_DEFINITIONS:
            score = 0
            if keyword_lower in api["short_name"].lower():
                score += 10
            if keyword_lower in api.get("summary", "").lower():
                score += 5
            for p in api.get("parameters", []):
                if keyword_lower in p.get("description", "").lower():
                    score += 2
            for p in api.get("body_params", []):
                if keyword_lower in p.get("description", "").lower():
                    score += 2
            if score > 0:
                results.append((score, api))

        results.sort(key=lambda x: -x[0])
        results = results[:limit]

        if not results:
            return f"未找到与 '{keyword}' 相关的 API。"

        lines = [f"搜索 '{keyword}' 的结果 ({len(results)} 个):\n"]
        for score, api in results:
            lines.append(f"  [{api['method']:4s}] {api['short_name']:40s} {api.get('summary', '')}")

        return "\n".join(lines)

    def _fuzzy_find(self, name: str, limit: int = 10) -> List[str]:
        """模糊匹配 API 名称"""
        name_lower = name.lower()
        results = []
        for api in API_DEFINITIONS:
            short = api["short_name"].lower()
            if name_lower in short:
                results.append(api["short_name"])
        return results[:limit]

    def _format_api_info(self, api: dict) -> str:
        """格式化 API 详情，展示位置参数用法"""
        lines = [
            f"  {api['short_name']}",
            f"    描述: {api.get('summary', 'N/A')}",
        ]

        ordered = self._get_ordered_params(api)
        if ordered:
            placeholders = []
            for p in ordered:
                ph = f"<{p['name']}>" if p.get("required") else f"[{p['name']}]"
                placeholders.append(ph)

            lines.append(f"    用法: /uapi {api['short_name']} {' '.join(placeholders)}")

            lines.append("\n    参数说明:")
            for p in ordered:
                pname = p["name"]
                req_str = "必填" if p.get("required") else "可选"
                ptype = p.get("type", "string")
                desc = p.get("description", "").strip()
                if len(desc) > 120:
                    desc = desc[:117] + "..."
                lines.append(f"    {pname} [{ptype}] ({req_str})")
                if desc:
                    lines.append(f"      {desc}")
                if p.get("enum"):
                    enum_vals = p["enum"]
                    if len(enum_vals) <= 8:
                        lines.append(f"      可选值: {', '.join(str(e) for e in enum_vals)}")
                    else:
                        lines.append(f"      可选值: {', '.join(str(e) for e in enum_vals[:6])} ... ({len(enum_vals)}个)")
        else:
            lines.append("    用法: /uapi " + api['short_name'] + "  (无需参数)")

        return "\n".join(lines)

    def _format_positional_usage(self, api: dict) -> str:
        """生成简短的参数用法说明（用于错误提示）"""
        ordered = self._get_ordered_params(api)
        if not ordered:
            return f"  /uapi {api['short_name']}  (此 API 无需参数)"

        placeholders = []
        for p in ordered:
            ph = f"<{p['name']}>" if p.get("required") else f"[{p['name']}]"
            placeholders.append(ph)

        usage = f"  /uapi {api['short_name']} {' '.join(placeholders)}\n"

        details = []
        for p in ordered:
            req_str = "必填" if p.get("required") else "可选"
            desc = p.get("description", "").strip()
            if len(desc) > 100:
                desc = desc[:97] + "..."
            details.append(f"    {p['name']} ({req_str}) {desc}")

        return usage + "\n" + "\n".join(details)


# ─────────────────────────────────────────────
# 动态 FunctionTool 工厂（供 LLM 调用）
# ─────────────────────────────────────────────

def _make_tool_instance(api: dict, client: UAPIClient):
    """为一个 API 定义动态创建 FunctionTool 实例。"""
    api_name = api["short_name"]
    summary = api.get("summary", api_name)
    method = api["method"]
    path = api["path"]
    query_params = api.get("parameters", [])
    body_params = api.get("body_params", [])

    # Build JSON Schema properties
    properties = {}
    required = []

    for p in query_params:
        prop: Dict[str, Any] = {
            "type": p.get("type", "string"),
            "description": p.get("description", ""),
        }
        if p.get("enum"):
            prop["enum"] = p["enum"]
        if p.get("default") is not None:
            prop["default"] = p["default"]
        properties[p["name"]] = prop
        if p.get("required"):
            required.append(p["name"])

    for p in body_params:
        prop: Dict[str, Any] = {
            "type": p.get("type", "string"),
            "description": p.get("description", ""),
        }
        properties[p["name"]] = prop
        if p.get("required"):
            required.append(p["name"])

    parameters_schema = {"type": "object", "properties": properties}
    if required:
        parameters_schema["required"] = required

    tool_name = f"uapi_{api_name.replace('.', '_')}"

    desc_lines = [f"{summary}（来源: UAPI, {method} {path}）"]
    if query_params:
        desc_lines.append(
            "查询参数: " + ", ".join(
                f"{p['name']}({'必填' if p.get('required') else '可选'})"
                for p in query_params[:5]
            )
        )
    if body_params:
        desc_lines.append(
            "请求体参数: " + ", ".join(
                f"{p['name']}({'必填' if p.get('required') else '可选'})"
                for p in body_params[:5]
            )
        )
    full_desc = "\n".join(desc_lines)

    _client = client
    _method = method
    _path = path
    _query_params = query_params
    _body_params = body_params
    _form_data = api.get("form_data", False)

    @dataclass
    class _DynamicTool(FunctionTool[AstrAgentContext]):
        name: str = tool_name
        description: str = full_desc
        parameters: dict = Field(default_factory=lambda: parameters_schema)

        async def call(
            self, context: ContextWrapper[AstrAgentContext], **kwargs
        ) -> ToolExecResult:
            """Execute the API call."""
            query_args = {}
            body_args = {}
            param_names = {p["name"] for p in _query_params}
            body_param_names = {p["name"] for p in _body_params}

            for key, value in kwargs.items():
                if key in ("self", "context", "cls"):
                    continue
                if key in param_names:
                    query_args[key] = value
                elif key in body_param_names:
                    body_args[key] = value
                elif _body_params:
                    body_args[key] = value
                else:
                    query_args[key] = value

            try:
                result = await _client.call(
                    path=_path,
                    method=_method,
                    params=query_args if query_args else None,
                    json_body=body_args if body_args and not _form_data else None,
                    form_data=body_args if body_args and _form_data else None,
                )

                if result.get("success"):
                    data = result.get("data", {})
                    if result.get("is_binary"):
                        return (
                            f"[UAPI {api_name}] 返回了二进制/图片数据 "
                            f"({len(result['data'])} bytes)"
                        )
                    if isinstance(data, (dict, list)):
                        formatted = json.dumps(data, ensure_ascii=False, indent=2)
                        if api_name == "misc.hotboard" and isinstance(data, dict):
                            items = data.get("list", data.get("results", []))
                            if isinstance(items, list):
                                platform = data.get("type", "")
                                update_time = data.get("update_time", "")
                                formatted = f"# {platform} 热榜"
                                if update_time:
                                    formatted += f"\n\n更新时间：{update_time}"
                                for position, item in enumerate(items, start=1):
                                    if not isinstance(item, dict):
                                        continue
                                    title = str(item.get("title", "未命名条目")).replace(
                                        "\n", " "
                                    )
                                    rank = item.get("index", position)
                                    formatted += f"\n\n## {rank}. {title}"
                                    if hot_value := item.get("hot_value"):
                                        formatted += f"\n热度：{hot_value}"
                                    if url := item.get("url"):
                                        formatted += f"\n链接：{url}"
                                try:
                                    image_path = await html_renderer.render_custom_template(
                                        PRESENTATION_TEMPLATE,
                                        {"content": formatted.replace("# ", "").replace("## ", "").replace("**", "")},
                                        return_url=False,
                                        options={"type": "png", "quality": 90},
                                    )
                                    event = context.context.event
                                    event.track_temporary_local_file(image_path)
                                    await event.send(
                                        MessageChain(
                                            chain=[Image.fromFileSystem(image_path)],
                                            type="tool_direct_result",
                                        )
                                    )
                                    return "热榜图片已发送给用户。"
                                except Exception as e:
                                    logger.warning(
                                        f"[UAPI] Failed to render hotboard result: {e}"
                                    )
                        if len(formatted) > 4000:
                            return (
                                f"[UAPI {api_name}] 返回内容过长，已截取前 4000 个字符。"
                                "请使用 /uapi 指令获取完整图片结果。\n\n"
                                + formatted[:4000]
                            )
                        return formatted
                    return str(data)
                else:
                    error_data = result.get("data")
                    error_msg = result.get("error")
                    if not error_msg and isinstance(error_data, dict):
                        error_msg = error_data.get("message") or error_data.get("code")
                    return f"[UAPI {api_name}] 调用失败: {error_msg or error_data or 'Unknown'}"
            except Exception as e:
                return f"[UAPI {api_name}] 异常: {str(e)}"

    return _DynamicTool()
