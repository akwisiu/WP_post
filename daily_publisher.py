"""
daily_publisher.py
------------------
每日自动发布脚本。执行流程：
1. 从任务队列中取出今天待发布的关键词
2. 联网搜索该关键词的最新新闻/热词
3. 调用 AI API 基于新闻素材写一篇符合SEO规范的文章
4. 对文章进行AI降率处理，降低AI检测率
5. 从 Pexels 搜索与关键词相关的图片，上传到 WordPress 媒体库
6. 发布文章（草稿或直接发布，根据配置决定）
7. 更新任务队列状态
"""

import json
import os
import re
import sys
import time
import hashlib
import requests
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
WP_CONFIG_FILE    = BASE_DIR / "wordpress_config.json"
API_CONFIG_FILE   = BASE_DIR / "api_config.json"
SEO_CONFIG_FILE   = BASE_DIR / "seo_prompt_config.json"
QUEUE_FILE        = BASE_DIR / "keyword_queue.json"
LOG_FILE          = BASE_DIR / "publish_log.jsonl"


# ──────────────────────────────────────────────
# 配置加载
# ──────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# 语言检测
# ──────────────────────────────────────────────

def detect_language(text: str) -> str:
    """返回 'en' 或 'zh'"""
    if not text:
        return "en"
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    alpha   = len(re.findall(r"[a-zA-Z\u4e00-\u9fff]", text))
    if alpha == 0:
        return "zh"
    return "zh" if chinese / alpha > 0.3 else "en"


# ──────────────────────────────────────────────
# Step 1: 联网搜索新闻/热词
# ──────────────────────────────────────────────

def search_news(keyword: str, seo_cfg: dict) -> str:
    """
    使用 DuckDuckGo Instant Answer API 搜索关键词相关摘要。
    返回拼接好的新闻背景文本供 AI 参考。
    """
    news_cfg = seo_cfg.get("news_search_config", {})
    queries  = news_cfg.get("search_queries", ["{keyword} news 2026"])
    max_items = news_cfg.get("max_news_items", 3)

    results = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ArticleBot/1.0)"}

    for tpl in queries[:2]:  # 最多查2个query，避免太慢
        query = tpl.replace("{keyword}", keyword)
        encoded = urllib.parse.quote(query)

        # DuckDuckGo Instant Answer（无需key，免费）
        try:
            url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                abstract = data.get("AbstractText", "")
                if abstract:
                    results.append(abstract)
                related = data.get("RelatedTopics", [])
                for item in related[:max_items]:
                    text = item.get("Text", "")
                    if text:
                        results.append(text)
        except Exception as e:
            print(f"  [search] DuckDuckGo query failed: {e}")

        # SerpAPI-free alternative: Brave Search (fallback，无需key)
        try:
            brave_url = f"https://search.brave.com/api/suggest?q={encoded}&rich=false"
            resp = requests.get(brave_url, headers=headers, timeout=8)
            if resp.status_code == 200:
                suggestions = resp.json()
                if isinstance(suggestions, list) and len(suggestions) > 1:
                    for s in suggestions[1][:3]:
                        if s:
                            results.append(str(s))
        except Exception:
            pass

        if len(results) >= max_items:
            break

    if results:
        combined = " | ".join(results[:max_items])
        print(f"  [search] Found {len(results)} results for '{keyword}'")
        return combined
    else:
        print(f"  [search] No results found for '{keyword}', proceeding without news context")
        return ""


# ──────────────────────────────────────────────
# Step 2: AI 写文章 (DeepSeek/Groq)
# ──────────────────────────────────────────────

def build_prompt(keyword: str, lang: str, news_context: str, seo_cfg: dict) -> str:
    """根据SEO配置构建写作提示词"""
    seo_req   = seo_cfg.get("seo_requirements", {})
    style_cfg = seo_cfg.get("writing_style", {})
    structure = seo_cfg.get("article_structure_template", {})
    lang_rule = seo_cfg.get("article_language_rule", "")

    title_rules   = "\n".join(f"  - {r}" for r in seo_req.get("title", []))
    content_rules = "\n".join(f"  - {r}" for r in seo_req.get("content_structure", []))
    meta_rules    = "\n".join(f"  - {r}" for r in seo_req.get("meta_description", []))
    read_rules    = "\n".join(f"  - {r}" for r in seo_req.get("readability", []))
    avoid_rules   = "\n".join(f"  - {r}" for r in style_cfg.get("avoid", []))
    tone          = style_cfg.get("tone", "professional")
    sections      = structure.get("sections", [])
    section_guide = "\n".join(
        f"  {i+1}. {s['name']}: {s['guidance']}"
        for i, s in enumerate(sections)
    )

    today = datetime.now().strftime("%B %d, %Y") if lang == "en" else datetime.now().strftime("%Y年%m月%d日")

    news_block = ""
    if news_context:
        if lang == "en":
            news_block = f"""
Latest news and trending information about "{keyword}" (collected today, {today}):
{news_context}

Use the above as background context and inspiration. Naturally integrate relevant facts or trends into the article. Do NOT copy the text verbatim.
"""
        else:
            news_block = f"""
关于"{keyword}"的今日最新资讯（{today}采集）：
{news_context}

请将上述内容作为写作背景和素材，自然融入文章，不要直接复制原文。
"""

    if lang == "en":
        prompt = f"""Write a SEO article about "{keyword}". Date: {today}.
{news_block}

Title: 50-60 chars, keyword at front.
Content: 800-1200 words, use h2 and h3 tags.
Meta: 150-160 chars.
Keyword density 1-2%, natural placement.
No emoji. Tone: {tone}.

Structure: Intro -> Background -> Core Analysis (2-3 subsections) -> Applications -> Conclusion.

Also provide 4 Pexels image search queries. Each query must be:
- A concrete, visual, real-world scene (e.g. "solar panels on rooftop", "battery storage facility interior")
- Directly related to the specific section content, NOT just the main keyword repeated
- 2-5 words, English only, no special characters
- Suitable for a professional stock photo search

JSON only:
{{"title":"", "meta_description":"", "content":"<h2>...</h2><p>...</p>", "image_queries":["query1","query2","query3","query4"]}}
"""
    else:
        prompt = f"""你是一位专业的SEO内容写作者。请围绕关键词"{keyword}"写一篇高质量、符合SEO规范的文章。

今天日期：{today}
{news_block}
语言规则：{lang_rule}

标题要求：
{title_rules}

内容要求：
{content_rules}

摘要要求：
{meta_rules}

可读性要求：
{read_rules}

避免：
{avoid_rules}

写作语气：{tone}

文章结构（按此顺序，使用 H2/H3 标签）：
{section_guide}

输出格式（只返回以下JSON，不要有其他文字）：
{{
  "title": "文章标题",
  "meta_description": "70-80字的SEO摘要",
  "content": "<h2>小标题</h2><p>内容...</p><h2>...</h2>...",
  "image_queries": ["英文搜索词1", "英文搜索词2", "英文搜索词3", "英文搜索词4"]
}}

image_queries 要求：
- 提供4个英文 Pexels 图片搜索词，分别对应文章各主要章节的核心视觉内容
- 每个搜索词必须是具体的、可视化的真实场景（如 "solar panels rooftop installation"、"battery storage warehouse"）
- 不要简单重复主关键词，每个词应针对对应章节的具体内容
- 2-5个英文单词，无特殊符号

重要说明：
- content 字段必须是合法HTML，使用 <h2>、<h3>、<p>、<ul>、<li> 标签
- 不要包含 <html>、<body>、<head> 标签
- content 字段中不要包含文章标题
- 不要使用任何 emoji 表情符号
- 正文至少 1000 字
"""
    return prompt


def write_article_with_deepseek(keyword: str, lang: str, news_context: str,
                                 seo_cfg: dict, api_cfg: dict) -> dict:
    """调用 AI API（支持 Groq/DeepSeek）生成文章"""
    provider = api_cfg.get("ai_provider", "deepseek")
    prompt = build_prompt(keyword, lang, news_context, seo_cfg)

    if provider == "groq":
        return _call_groq(prompt, keyword, api_cfg)
    elif provider == "deepseek":
        return _call_deepseek(prompt, keyword, seo_cfg, api_cfg)
    else:
        raise ValueError(f"Unknown AI provider: {provider}")


def _call_deepseek(prompt: str, keyword: str, seo_cfg: dict, api_cfg: dict) -> dict:
    """调用 DeepSeek API"""
    ds_cfg   = seo_cfg.get("deepseek_config", {})
    model    = ds_cfg.get("model", "deepseek-chat")
    temp     = ds_cfg.get("temperature", 0.7)
    max_tok  = ds_cfg.get("max_tokens", 3000)
    api_base = ds_cfg.get("api_base", "https://api.deepseek.com/v1")
    api_key  = api_cfg.get("deepseek_api_key", "")

    if not api_key or api_key.startswith("sk-") and len(api_key) < 20:
        raise ValueError("DeepSeek API key not set in api_config.json")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model":       model,
        "temperature": temp,
        "max_tokens":  max_tok,
        "messages": [
            {"role": "system", "content": "You are a professional SEO content writer. Always respond in valid JSON format as instructed."},
            {"role": "user",   "content": prompt}
        ]
    }

    print(f"  [deepseek] Calling API for keyword: {keyword}")
    resp = requests.post(f"{api_base}/chat/completions",
                         headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*",     "", raw)
    raw = re.sub(r"\s*```$",     "", raw)
    article = json.loads(raw)
    print(f"  [deepseek] Article generated: {article.get('title', '')[:60]}")
    return article


def _call_groq(prompt: str, keyword: str, api_cfg: dict) -> dict:
    """调用 Groq API（兼容 OpenAI 格式），修复控制字符问题"""
    model   = api_cfg.get("groq_model", "llama-3.1-70b-versatile")
    api_key = api_cfg.get("groq_api_key", "")

    if not api_key or api_key.startswith("gsk_your"):
        raise ValueError("Groq API key not set in api_config.json")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model":       model,
        "temperature": 0.7,
        "max_tokens":  3000,
        "messages": [
            {"role": "system", "content": "You are a professional SEO content writer. Always respond in valid JSON format as instructed."},
            {"role": "user",   "content": prompt}
        ]
    }

    print(f"  [groq] Calling API for keyword: {keyword}")
    print(f"  [groq] Model: {model}, prompt length: {len(prompt)} chars")
    resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                         headers=headers, json=payload, timeout=120)
    if not resp.ok:
        error_detail = resp.text
        print(f"  [groq] Error response: {error_detail}")
    resp.raise_for_status()

    result = resp.json()
    print(f"  [groq] Response keys: {result.keys()}")

    if "choices" not in result or not result["choices"]:
        print(f"  [groq] No choices in response: {result}")
        raise ValueError("Groq API returned no choices, full response: " + str(result))

    raw = result["choices"][0]["message"]["content"].strip()
    print(f"  [groq] Raw response (first 500 chars): {raw[:500]}")
    print(f"  [groq] Raw response length: {len(raw)} chars")

    # 第一步：清理代码块标记
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*",     "", raw)
    raw = re.sub(r"\s*```$",     "", raw)
    raw = raw.strip()
    
    print(f"  [groq] After code block cleanup: {len(raw)} chars")
    
    # 尝试修复 JSON 中的控制字符问题
    # Groq API 可能在字符串中返回了真正的控制字符（如换行符），
    # 但在 JSON 中这些字符必须被转义
    
    def fix_json_control_chars(json_str: str) -> str:
        """修复 JSON 字符串中的控制字符"""
        # 首先尝试直接解析，可能会失败
        try:
            temp_obj = json.loads(json_str)
            # 如果成功，重新序列化以确保正确的格式
            return json.dumps(temp_obj)
        except json.JSONDecodeError:
            pass
        
        # JSON 解析失败，尝试修复
        # 查找字符串中的控制字符并转义它们
        import string
        
        # 将控制字符转义为 \uXXXX 格式
        def escape_control_char(match):
            char = match.group(0)
            code = ord(char)
            if char == '\n':
                return '\\n'
            elif char == '\r':
                return '\\r'
            elif char == '\t':
                return '\\t'
            elif char == '\b':
                return '\\b'
            elif char == '\f':
                return '\\f'
            elif code < 32:
                return f'\\u{code:04x}'
            return char
        
        # 只在双引号字符串内部进行转义
        result = []
        i = 0
        in_string = False
        escape_next = False
        
        while i < len(json_str):
            ch = json_str[i]
            
            if escape_next:
                result.append(ch)
                escape_next = False
                i += 1
                continue
            
            if ch == '\\':
                result.append(ch)
                escape_next = True
                i += 1
                continue
            
            if ch == '"':
                # 检查是否是转义的双引号
                if i > 0 and json_str[i-1] == '\\':
                    result.append(ch)
                else:
                    result.append(ch)
                    in_string = not in_string
                i += 1
                continue
            
            if in_string and ord(ch) < 32:
                # 在字符串中的控制字符，需要转义
                code = ord(ch)
                if ch == '\n':
                    result.append('\\n')
                elif ch == '\r':
                    result.append('\\r')
                elif ch == '\t':
                    result.append('\\t')
                elif code < 32:
                    result.append(f'\\u{code:04x}')
            else:
                result.append(ch)
            
            i += 1
        
        fixed = ''.join(result)
        print(f"  [groq] Fixed JSON length: {len(fixed)} chars")
        return fixed
    
    try:
        # 尝试修复控制字符
        fixed_json = fix_json_control_chars(raw)
        
        print(f"  [groq] First 300 chars of fixed JSON: {fixed_json[:300]}")
        
        try:
            article = json.loads(fixed_json)
            print(f"  [groq] JSON解析成功!")
        except json.JSONDecodeError as e:
            print(f"  [groq] 修复后仍然解析失败: {e}")
            print(f"  [groq] 尝试简单清理方法...")
            
            # 使用最简单的清理方法：只保留可打印字符
            import string
            printable = set(string.printable)
            simple_clean = ''.join(filter(lambda x: x in printable, raw))
            
            # 尝试解析简单清理后的数据
            try:
                article = json.loads(simple_clean)
                print(f"  [groq] 简单清理后解析成功")
            except json.JSONDecodeError as e_simple:
                print(f"  [groq] 简单清理也失败: {e_simple}")
                raise
    
    except Exception as e:
        print(f"  [groq] JSON 处理失败: {e}")
        raise
    
    print(f"  [groq] Article generated: {article.get('title', '')[:60]}")
    return article


# ──────────────────────────────────────────────
# Step 4: Pexels 搜索配图
# ──────────────────────────────────────────────

def count_words(html_content: str) -> int:
    """统计HTML内容的词数（英文按空格分词，中文按字符数）"""
    text = re.sub(r"<[^>]+>", "", html_content)
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_words = len(text.split())
    return chinese + english_words


def get_image_count(word_count: int, seo_cfg: dict) -> int:
    """根据字数确定配图数量"""
    rules = seo_cfg.get("pexels_config", {}).get("image_count_rules", {})
    if word_count < 300:
        return rules.get("under_300_words", 1)
    elif word_count < 700:
        return rules.get("300_to_700_words", 2)
    elif word_count < 1500:
        return rules.get("700_to_1500_words", 3)
    else:
        return rules.get("over_1500_words", 4)


def search_pexels_images(keyword: str, count: int, api_key: str,
                          orientation: str = "landscape") -> list:
    """
    从 Pexels 搜索与关键词相关的图片。
    返回图片信息列表：[{"url": ..., "photographer": ..., "alt": ...}, ...]
    """
    if not api_key:
        print("  [pexels] No API key, falling back to picsum")
        return _picsum_fallback(keyword, count)

    headers = {"Authorization": api_key}
    params  = {
        "query":       keyword,
        "orientation": orientation,
        "per_page":    max(count + 2, 5),  # 多搜一些，再筛选
        "page":        1
    }

    try:
        resp = requests.get("https://api.pexels.com/v1/search",
                            headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        photos = resp.json().get("photos", [])
        if not photos:
            print(f"  [pexels] No results for '{keyword}', trying fallback query")
            # 尝试用关键词的第一个单词重新搜索
            fallback_kw = keyword.split()[0] if " " in keyword else keyword
            params["query"] = fallback_kw
            resp = requests.get("https://api.pexels.com/v1/search",
                                headers=headers, params=params, timeout=15)
            photos = resp.json().get("photos", [])

        results = []
        for photo in photos[:count]:
            results.append({
                "url":          photo["src"]["large"],
                "url_original": photo["src"]["original"],
                "photographer": photo.get("photographer", "Pexels"),
                "alt":          photo.get("alt", keyword),
                "pexels_id":    photo["id"]
            })
        print(f"  [pexels] Found {len(results)} images for '{keyword}'")
        return results
    except Exception as e:
        print(f"  [pexels] Error: {e}, falling back to picsum")
        return _picsum_fallback(keyword, count)


def _picsum_fallback(keyword: str, count: int) -> list:
    """Pexels 失败时的备用图片"""
    base_seed = int(hashlib.md5(keyword.encode()).hexdigest(), 16) % 1000
    results = []
    for i in range(count):
        seed = (base_seed + i * 37) % 1000
        results.append({
            "url":          f"https://picsum.photos/seed/{seed}/800/500",
            "url_original": f"https://picsum.photos/seed/{seed}/1200/630",
            "photographer": "Picsum",
            "alt":          keyword,
            "pexels_id":    None
        })
    return results


# ──────────────────────────────────────────────
# Step 5: 上传图片到 WordPress 媒体库
# ──────────────────────────────────────────────

def download_image(url: str, timeout: int = 20) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def upload_image_to_wordpress(image_data: bytes, filename: str, alt_text: str,
                               wp_cfg: dict) -> Optional[int]:
    """上传图片到 WordPress，返回 media_id"""
    url   = wp_cfg["wordpress_url"].rstrip("/") + "/wp-json/wp/v2/media"
    auth  = (wp_cfg["username"], wp_cfg["app_password"])
    ext   = filename.rsplit(".", 1)[-1] if "." in filename else "jpg"
    mime  = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
             "png": "image/png",  "webp": "image/webp"}.get(ext, "image/jpeg")

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type":        mime
    }
    params = {"alt_text": alt_text, "caption": alt_text}

    try:
        resp = requests.post(url, auth=auth, headers=headers,
                             data=image_data, params=params,
                             timeout=60, verify=False)
        resp.raise_for_status()
        media = resp.json()
        media_id = media.get("id")
        media_url = media.get("source_url", "")
        print(f"  [wp_media] Uploaded '{filename}' -> media_id={media_id}")
        return media_id, media_url
    except Exception as e:
        print(f"  [wp_media] Upload failed for '{filename}': {e}")
        return None, None


# ──────────────────────────────────────────────
# Step 6: 将配图插入文章正文
# ──────────────────────────────────────────────

def insert_images_into_content(content: str, image_urls: list,
                                image_alts: list) -> str:
    """在文章 H2 标题前均匀插入配图"""
    if not image_urls:
        return content

    h2_positions = [m.start() for m in re.finditer(r"<h2", content, re.IGNORECASE)]

    if len(h2_positions) < 2:
        # 没有足够的H2，直接在开头后插入
        img_tags = "".join(
            f'<figure class="wp-block-image"><img src="{url}" alt="{alt}" style="max-width:100%;height:auto;margin:20px 0;" /></figure>\n'
            for url, alt in zip(image_urls, image_alts)
        )
        return img_tags + content

    # 均匀分布在各个H2之前
    step = max(1, len(h2_positions) // len(image_urls))
    insert_at = [h2_positions[min(i * step, len(h2_positions) - 1)]
                 for i in range(len(image_urls))]
    insert_at = sorted(set(insert_at), reverse=True)

    for idx, pos in enumerate(insert_at):
        img_idx = len(insert_at) - 1 - idx
        if img_idx < len(image_urls):
            img_tag = (
                f'\n<figure class="wp-block-image">'
                f'<img src="{image_urls[img_idx]}" alt="{image_alts[img_idx]}" '
                f'style="max-width:100%;height:auto;margin:20px 0;" />'
                f'</figure>\n'
            )
            content = content[:pos] + img_tag + content[pos:]

    return content


# ──────────────────────────────────────────────
# Step 7: 发布到 WordPress
# ──────────────────────────────────────────────

def get_or_create_category(category_name: str, wp_cfg: dict) -> Optional[int]:
    """获取或创建 WordPress 分类，返回分类ID"""
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    if not category_name or category_name.strip() == "":
        return None
    
    base_url = wp_cfg["wordpress_url"].rstrip("/")
    auth = (wp_cfg["username"], wp_cfg["app_password"])
    
    # 1. 首先尝试查找现有分类
    search_url = f"{base_url}/wp-json/wp/v2/categories?search={requests.utils.quote(category_name)}"
    try:
        resp = requests.get(search_url, auth=auth, timeout=30, verify=False)
        if resp.status_code == 200:
            categories = resp.json()
            for cat in categories:
                if cat.get("name", "").lower() == category_name.lower() or cat.get("slug", "").lower() == category_name.lower().replace(" ", "-"):
                    return cat.get("id")
    except Exception as e:
        print(f"  [category] Search error: {e}")
    
    # 2. 如果没有找到，尝试创建新分类
    create_url = f"{base_url}/wp-json/wp/v2/categories"
    try:
        slug = category_name.lower().replace(" ", "-")
        create_data = {
            "name": category_name,
            "slug": slug
        }
        resp = requests.post(create_url, auth=auth, json=create_data, timeout=30, verify=False)
        if resp.status_code in [200, 201]:
            category = resp.json()
            print(f"  [category] Created new category: {category_name} (ID: {category.get('id')})")
            return category.get("id")
        else:
            print(f"  [category] Failed to create category '{category_name}': {resp.status_code}")
    except Exception as e:
        print(f"  [category] Creation error: {e}")
    
    return None


def publish_to_wordpress(title: str, content: str, excerpt: str,
                          featured_media_id: Optional[int],
                          keyword: str, category: str, wp_cfg: dict) -> dict:
    """发布文章到 WordPress"""
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    url   = wp_cfg["wordpress_url"].rstrip("/") + "/wp-json/wp/v2/posts"
    auth  = (wp_cfg["username"], wp_cfg["app_password"])
    status = "draft" if wp_cfg.get("draft_mode", True) else "publish"

    # 获取或创建分类
    category_id = None
    if category and category.strip():
        category_id = get_or_create_category(category, wp_cfg)
        if category_id:
            print(f"  [wordpress] Using category: {category} (ID: {category_id})")
        else:
            print(f"  [wordpress] Category '{category}' not available, will use default")

    post_data = {
        "title":   title,
        "content": content,
        "excerpt": excerpt,
        "status":  status,
    }
    if featured_media_id:
        post_data["featured_media"] = featured_media_id
    if category_id:
        post_data["categories"] = [category_id]

    resp = requests.post(url, auth=auth, json=post_data,
                         timeout=60, verify=False)
    resp.raise_for_status()
    post = resp.json()
    post_id  = post.get("id")
    post_url = post.get("link", "")
    print(f"  [wordpress] Published post_id={post_id}, status={status}")
    print(f"  [wordpress] URL: {post_url}")
    return {"post_id": post_id, "post_url": post_url, "status": status}


# ──────────────────────────────────────────────
# 任务队列管理
# ──────────────────────────────────────────────

def load_queue() -> dict:
    if QUEUE_FILE.exists():
        return load_json(QUEUE_FILE)
    return {"keywords": [], "completed": [], "failed": []}


def get_next_keyword(queue: dict) -> Optional[dict]:
    """获取下一个待发布的关键词"""
    completed_kws = {item["keyword"] for item in queue.get("completed", [])}
    failed_kws    = {item["keyword"] for item in queue.get("failed", [])
                     if item.get("retry_count", 0) >= 3}  # 失败3次后跳过

    for item in queue.get("keywords", []):
        kw = item["keyword"] if isinstance(item, dict) else item
        if kw not in completed_kws and kw not in failed_kws:
            return item if isinstance(item, dict) else {"keyword": kw}
    return None


def mark_completed(queue: dict, keyword: str, result: dict) -> None:
    # 从keywords列表中移除已完成的关键词
    if "keywords" in queue:
        # 找到并移除匹配的关键词
        for i, item in enumerate(queue["keywords"]):
            kw = item["keyword"] if isinstance(item, dict) else item
            if kw == keyword:
                queue["keywords"].pop(i)
                break
    
    # 添加到completed列表
    queue.setdefault("completed", []).append({
        "keyword":    keyword,
        "completed_at": datetime.now().isoformat(),
        "post_id":    result.get("post_id"),
        "post_url":   result.get("post_url"),
        "status":     result.get("status")
    })
    save_json(QUEUE_FILE, queue)


def mark_failed(queue: dict, keyword: str, error: str) -> None:
    failed = queue.setdefault("failed", [])
    for item in failed:
        if item["keyword"] == keyword:
            item["retry_count"] = item.get("retry_count", 0) + 1
            item["last_error"]  = error
            item["last_attempt"] = datetime.now().isoformat()
            save_json(QUEUE_FILE, queue)
            return
    failed.append({
        "keyword":     keyword,
        "retry_count": 1,
        "last_error":  error,
        "last_attempt": datetime.now().isoformat()
    })
    save_json(QUEUE_FILE, queue)


def append_log(entry: dict) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def run_daily_publish():
    print("=" * 60)
    print(f"Daily Publisher - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 加载配置
    try:
        wp_cfg  = load_json(WP_CONFIG_FILE)
        api_cfg = load_json(API_CONFIG_FILE)
        seo_cfg = load_json(SEO_CONFIG_FILE)
    except FileNotFoundError as e:
        print(f"Config file missing: {e}")
        sys.exit(1)

    # 加载任务队列
    queue = load_queue()
    item  = get_next_keyword(queue)

    if not item:
        print("No pending keywords in queue. All done or queue is empty.")
        print("Add keywords to keyword_queue.json to continue.")
        return

    keyword  = item["keyword"]
    category = item.get("category", "")
    lang     = detect_language(keyword)

    print(f"\nKeyword  : {keyword}")
    print(f"Language : {lang}")
    print(f"Category : {category}")
    print(f"Remaining: {len([k for k in queue.get('keywords',[]) if (k['keyword'] if isinstance(k,dict) else k) not in {x['keyword'] for x in queue.get('completed',[])}])} keywords")
    print("-" * 60)

    try:
        # Step 1: 联网搜索新闻
        print("\n[Step 1] Searching for latest news...")
        news_context = search_news(keyword, seo_cfg)

        # Step 2: AI 写文章
        print("\n[Step 2] Writing article with AI...")
        article = write_article_with_deepseek(keyword, lang, news_context, seo_cfg, api_cfg)
        title    = article.get("title", keyword)
        content  = article.get("content", "")
        excerpt  = article.get("meta_description", "")

        # Step 3: AI降率处理
        print("\n[Step 3] Reducing AI detection rate...")
        try:
            # 导入humano_integration模块
            from humano_integration import reduce_ai_percentage
            original_content = content
            content = reduce_ai_percentage(content, keyword, strength="medium")
            
            # 计算AI降率处理的修改比例（简化版）
            if original_content != content:
                print("  AI降率处理完成：内容已进行人性化优化")
            else:
                print("  AI降率处理完成：内容无变化或该语言类型不支持")
        except ImportError as e:
            print(f"  警告: Humano集成模块未找到 - {e}")
            print("  跳过AI降率功能，继续下一步")
        except Exception as e:
            print(f"  AI降率处理出错: {e}")
            print("  跳过此步骤，继续执行")

        # Step 4: 搜索 Pexels 配图
        print("\n[Step 4] Fetching images from Pexels...")
        word_count  = count_words(content)
        img_count   = get_image_count(word_count, seo_cfg)
        pexels_key  = api_cfg.get("pexels_api_key", "")

        # 优先使用 AI 生成的精准图片搜索词
        image_queries = article.get("image_queries", [])
        if image_queries:
            print(f"  [pexels] Using AI-generated image queries: {image_queries}")
        else:
            print(f"  [pexels] No image_queries from AI, falling back to keyword")

        # 特色图：用第一个 image_query，没有则用 keyword
        featured_query = image_queries[0] if image_queries else keyword
        featured_images = search_pexels_images(featured_query, 1, pexels_key, "landscape")

        # 内联图：每张用对应的 image_query，超出范围则循环使用或退回 keyword
        inline_images = []
        for i in range(img_count):
            if image_queries and i + 1 < len(image_queries):
                q = image_queries[i + 1]
            elif image_queries:
                q = image_queries[i % len(image_queries)]
            else:
                q = keyword
            imgs = search_pexels_images(q, 1, pexels_key, "landscape")
            if imgs:
                inline_images.append(imgs[0])
            else:
                # 最终 fallback 用 keyword
                fallback = search_pexels_images(keyword, 1, pexels_key, "landscape")
                if fallback:
                    inline_images.append(fallback[0])

        print(f"  Word count: {word_count}, inline images: {img_count}")

        # Step 5: 上传特色图片到 WordPress
        print("\n[Step 5] Uploading featured image to WordPress...")
        featured_media_id = None
        if featured_images:
            fi = featured_images[0]
            try:
                img_data = download_image(fi["url_original"])
                safe_name = re.sub(r"[^a-z0-9]", "-", keyword.lower())[:40]
                filename  = f"{safe_name}-featured.jpg"
                featured_media_id, _ = upload_image_to_wordpress(
                    img_data, filename, fi["alt"], wp_cfg
                )
            except Exception as e:
                print(f"  [featured] Image upload failed: {e}")

        # Step 6: 上传配图并插入正文
        print("\n[Step 6] Uploading inline images and inserting into content...")
        inline_urls = []
        inline_alts = []
        for idx, img in enumerate(inline_images):
            try:
                img_data = download_image(img["url"])
                safe_name = re.sub(r"[^a-z0-9]", "-", keyword.lower())[:30]
                filename  = f"{safe_name}-inline-{idx+1}.jpg"
                _, wp_url = upload_image_to_wordpress(
                    img_data, filename, img["alt"], wp_cfg
                )
                if wp_url:
                    inline_urls.append(wp_url)
                    inline_alts.append(img["alt"])
                else:
                    # 上传失败则直接用 Pexels URL
                    inline_urls.append(img["url"])
                    inline_alts.append(img["alt"])
            except Exception as e:
                print(f"  [inline] Image {idx+1} failed: {e}")
                inline_urls.append(img["url"])
                inline_alts.append(img["alt"])

            time.sleep(1)  # 避免请求过快

        content_with_images = insert_images_into_content(content, inline_urls, inline_alts)

        # Step 7: 发布到 WordPress
        print("\n[Step 7] Publishing to WordPress...")
        result = publish_to_wordpress(
            title, content_with_images, excerpt,
            featured_media_id, keyword, category, wp_cfg
        )

        # 记录成功
        mark_completed(queue, keyword, result)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "keyword":   keyword,
            "language":  lang,
            "title":     title,
            "word_count": word_count,
            "images":    img_count + 1,
            "post_id":   result["post_id"],
            "post_url":  result["post_url"],
            "status":    result["status"],
            "success":   True
        }
        append_log(log_entry)

        print("\n" + "=" * 60)
        print("SUCCESS")
        print(f"  Title   : {title}")
        print(f"  Words   : {word_count}")
        print(f"  Images  : {img_count + 1} (1 featured + {img_count} inline)")
        print(f"  Post ID : {result['post_id']}")
        print(f"  URL     : {result['post_url']}")
        print(f"  Status  : {result['status']}")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        mark_failed(queue, keyword, str(e))
        append_log({
            "timestamp": datetime.now().isoformat(),
            "keyword":   keyword,
            "success":   False,
            "error":     str(e)
        })
        sys.exit(1)


if __name__ == "__main__":
    run_daily_publish()
