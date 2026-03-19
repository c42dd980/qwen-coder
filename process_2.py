import asyncio
import datetime
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import urllib3
from playwright.async_api import BrowserContext, Locator, Page, async_playwright

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ⚙️ Настройки
MAX_CONCURRENT = 4
MAX_RETRIES = 3
RETRY_DELAY = 2
BATCH_SIZE = 50
BATCH_PAUSE = 10
PAGINATION_WAIT = 3000
VALID_EXTENSIONS: set[str] = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".txt",
    ".rtf",
    ".zip",
    ".rar",
    ".7z",
    ".mov",
    ".mp4",
    ".avi",
    ".jpg",
    ".png",
}

# 📝 Логирование
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data/download.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# 🔢 Паттерн номера заказа: 7 цифр - 6 цифр (напр. 2709025-171677)
ORDER_NUM_PATTERN = re.compile(r"\b(\d{7}-\d{6})\b")


def get_today_folder() -> str:
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    base = f"data/download/{today}"
    Path(f"{base}/appeals").mkdir(parents=True, exist_ok=True)
    return base


def sanitize_filename(filename: str) -> str:
    if not filename:
        return ""
    return "".join(c for c in filename if c.isalnum() or c in "._- ").strip()[:150]


def normalize_url(url: Optional[str], base: str = "https://") -> Optional[str]:
    if not url:
        return None
    url = url.strip()
    if url.startswith("//"):
        return base + url.lstrip("/")
    return url


def is_valid_file(filepath: str) -> bool:
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return False
    signatures: List[tuple[bytes, List[str]]] = [
        (b"%PDF-", [".pdf"]),
        (b"\xd0\xcf\x11\xe0", [".doc", ".xls", ".ppt"]),
        (b"PK\x03\x04", [".docx", ".xlsx", ".pptx", ".zip"]),
        (b"ftyp", [".mov", ".mp4", ".m4v"]),
        (b"\xff\xd8\xff", [".jpg", ".jpeg"]),
        (b"\x89PNG\r\n\x1a\n", [".png"]),
    ]
    ext = Path(filepath).suffix.lower()
    try:
        with open(filepath, "rb") as f:
            header = f.read(16)
        for sig, exts in signatures:
            if header.startswith(sig) and ext in exts:
                return True
        if ext in VALID_EXTENSIONS and os.path.getsize(filepath) > 100:
            return True
        if ext in {".txt", ".csv", ".xml", ".json"} and os.path.getsize(filepath) > 0:
            return True
        return False
    except Exception:
        return False


def download_with_auth(
    cookies: dict,
    user_agent: str,
    referer: str,
    url: str,
    dest_path: str,
    verify_ssl: bool = False,
) -> bool:
    for attempt in range(MAX_RETRIES):
        try:
            session = requests.Session()
            for name, value in cookies.items():
                session.cookies.set(name, value, domain="", path="/")
            headers = {"User-Agent": user_agent, "Referer": referer}
            response = session.get(
                url, stream=True, timeout=30, headers=headers, verify=verify_ssl
            )
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            ext = Path(dest_path).suffix.lower()
            if "text/html" in content_type and ext not in VALID_EXTENSIONS:
                logger.warning(f"⚠ Suspicious response for {url}")
                return False
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if not is_valid_file(dest_path):
                logger.warning(f"✗ Invalid file content: {dest_path}")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                return False
            return True
        except Exception as e:
            logger.warning(f"✗ Download error (attempt {attempt + 1}): {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))
    return False


async def wait_for_page_ready(page: Page, timeout_ms: int = 10000):
    await page.wait_for_load_state("networkidle", timeout=timeout_ms)
    await page.wait_for_function(
        "() => document.readyState === 'complete'", timeout=timeout_ms
    )


async def extract_cookies_for_requests(context: BrowserContext, base_url: str) -> dict:
    cookies = await context.cookies()
    result = {}
    base_domain = base_url.replace("https://", "").replace("http://", "").split("/")[0]
    for c in cookies:
        domain = c.get("domain", "").lstrip(".")
        if not domain or domain in base_domain or base_domain.endswith(domain):
            result[c["name"]] = c["value"]
    return result


async def extract_order_number(page: Page) -> Optional[str]:
    """Извлекает номер заказа из textarea#commentText по паттерну \\d{7}-\\d{6}"""
    try:
        textarea = page.locator("textarea#commentText").first
        if await textarea.count() == 0:
            return None
        text = await textarea.text_content() or ""
        match = ORDER_NUM_PATTERN.search(text)
        return match.group(1) if match else None
    except Exception as e:
        logger.debug(f"⚠ Error extracting order number: {e}")
        return None


async def parse_results_table_with_pagination(
    page: Page,
) -> List[Dict[str, Optional[str]]]:
    """Парсит таблицу с полной пагинацией"""
    records = []
    seen_appeals = set()
    appeal_pattern = re.compile(r"270-\d{8}")
    date_pattern = re.compile(r"\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}")
    phone_pattern = re.compile(r"(\+?7\d{10,11})")
    email_pattern = re.compile(r"([\w\.-]+@[\w\.-]+\.\w+)")

    logger.info("🔍 Starting table parsing with pagination...")
    try:
        await page.wait_for_selector('a[href*="/appeals/1form/"]', timeout=60000)
    except Exception as e:
        logger.error(f"✗ Table not found: {e}")
        return records

    async def parse_row(row_locator: Locator) -> Optional[Dict]:
        try:
            appeal_link_el = row_locator.locator('a[href*="/appeals/1form/"]').first
            href = await appeal_link_el.get_attribute("href")
            appeal_match = appeal_pattern.search(href or "")
            if not appeal_match:
                return None
            appeal_num = appeal_match.group()
            if appeal_num in seen_appeals:
                return None
            seen_appeals.add(appeal_num)

            appeal_link = normalize_url(href)
            row_text = await row_locator.text_content() or ""

            created_at = None
            date_matches = date_pattern.findall(row_text)
            if date_matches:
                created_at = date_matches[0]

            topic_short = None
            if "Запрос документов от РС" in row_text:
                topic_short = "Запрос документов от РС"
            elif "HUMAN HELP" in row_text:
                topic_short = "HUMAN HELP"

            topic_full = None
            topic_match = re.search(
                r"(HUMAN HELP/[^\n]+?)(?:Закрыто|Определен исполнитель|В работе|Открыто|Решено)",
                row_text,
            )
            if topic_match:
                topic_full = topic_match.group(1).strip()

            status = None
            for kw in [
                "Закрыто",
                "Определен исполнитель",
                "В работе",
                "Открыто",
                "Решено",
                "Незакрытые",
            ]:
                if kw in row_text:
                    status = kw
                    break

            closed_at = None
            if status == "Закрыто":
                closed_match = re.search(
                    r"Закрыто\s*\n?\s*(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2})", row_text
                )
                if closed_match:
                    closed_at = closed_match.group(1)

            contact_name = None
            try:
                contact_el = row_locator.locator('a[href*="javascript:void"]').first
                if await contact_el.count() > 0:
                    contact_name = (await contact_el.text_content() or "").strip()
            except:
                pass

            contact_phone = contact_email = None
            phone_match = phone_pattern.search(row_text)
            email_match = email_pattern.search(row_text)
            if phone_match:
                contact_phone = phone_match.group(1)
            if email_match:
                contact_email = email_match.group(1)

            return {
                "appeal_num": appeal_num,
                "appeal_link": appeal_link,
                "created_at": created_at,
                "topic_short": topic_short,
                "topic_full": topic_full,
                "status": status,
                "closed_at": closed_at,
                "contact_name": contact_name,
                "contact_phone": contact_phone,
                "contact_email": contact_email,
                "order_num": None,  # ← Заполнится при обработке страницы
            }
        except Exception as e:
            logger.debug(f"⚠ Error parsing row: {e}")
            return None

    async def extract_current_page() -> int:
        nonlocal records
        appeal_links = await page.locator('a[href*="/appeals/1form/"]').all()
        new_count = 0
        for link in appeal_links:
            parent = link.locator(
                "xpath=ancestor::div[contains(@class, 'row') or contains(@class, 'item') or @role='row'][1]"
            )
            if await parent.count() == 0:
                parent = link.locator("xpath=ancestor::div[1]")
            data = await parse_row(parent)
            if data:
                records.append(data)
                new_count += 1
        return new_count

    first_count = await extract_current_page()
    logger.info(f"✓ Initial page: {first_count} records, total: {len(records)}")

    iteration = 0
    max_iterations = 100
    consecutive_no_new = 0

    while iteration < max_iterations:
        iteration += 1
        logger.info(f"🔄 Pagination iteration {iteration}/{max_iterations}")
        before_count = len(records)

        show_more_selectors = [
            'button:has-text("Загрузить ещё")',
            'a:has-text("Загрузить ещё")',
            'button:has-text("Загрузить еще")',
            'a:has-text("Загрузить еще")',
            'button:has-text("Показать еще")',
            'a:has-text("Показать еще")',
            'button:has-text("Показать ещё")',
            'a:has-text("Показать ещё")',
            '[role="button"]:has-text("Загрузить")',
            '[role="button"]:has-text("Показать")',
        ]
        show_more = None
        for selector in show_more_selectors:
            try:
                locator = page.locator(selector).first
                if await locator.count() > 0 and await locator.is_visible(timeout=2000):
                    show_more = locator
                    break
            except:
                continue

        if not show_more:
            logger.info("✅ No more 'Show more' button - pagination complete")
            break

        try:
            await show_more.click(timeout=10000)
        except Exception as e:
            logger.warning(f"⚠ Click failed: {e}")
            try:
                await show_more.evaluate("el => el.click()")
            except:
                break

        await page.wait_for_timeout(PAGINATION_WAIT)
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except:
            pass

        new_added = await extract_current_page()
        logger.info(f"📊 Added {new_added} new records, total: {len(records)}")

        if new_added == 0:
            consecutive_no_new += 1
            if consecutive_no_new >= 3:
                logger.info("✅ 3 iterations without new data - stopping")
                break
        else:
            consecutive_no_new = 0

        try:
            counter = await page.locator('text="Показано"').first.text_content(
                timeout=2000
            )
            match = re.search(r"Показано\s+(\d+)\s+из\s+(\d+)", counter or "")
            if match and int(match.group(1)) >= int(match.group(2)):
                logger.info("✅ All records loaded")
                break
        except:
            pass

    logger.info(f"🎉 Pagination complete: {len(records)} total records")
    return records


def save_to_excel(records: List[Dict], folder: str) -> str:
    if not records:
        logger.warning("⚠ No records to save")
        return ""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    filepath = f"{folder}/report_{today}.xlsx"
    df = pd.DataFrame(records)
    cols_order = [
        "appeal_num",
        "appeal_link",
        "created_at",
        "closed_at",
        "status",
        "topic_short",
        "topic_full",
        "contact_name",
        "contact_phone",
        "contact_email",
        "order_num",  # ← добавлен
    ]
    df = df[[c for c in cols_order if c in df.columns]]
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Обращения")
    logger.info(f"💾 Saved Excel report: {filepath}")
    return filepath


async def process_appeal_page(
    page: Page,
    context: BrowserContext,
    appeal_num: str,
    base_appeal_url: str,
    output_dir: str,
    auth_url: str,
) -> tuple[bool, Optional[str]]:
    """Обрабатывает страницу обращения, возвращает (успех, order_num)"""
    try:
        appeal_url = base_appeal_url.format(appeal_num)
        await page.goto(appeal_url, wait_until="domcontentloaded", timeout=30000)
        await wait_for_page_ready(page)
        await asyncio.sleep(1)

        # 🔍 Извлекаем номер заказа ДО работы с файлами
        order_num = await extract_order_number(page)
        if order_num:
            logger.info(f"[{appeal_num}] 🧾 Order found: {order_num}")
        else:
            logger.debug(f"[{appeal_num}] ⚠ Order number not found in #commentText")

        # 📎 Работа с файлами (без изменений)
        try:
            await page.wait_for_selector("div#filesList", timeout=15000)
            files_locator: Locator = page.locator("div#filesList a")
        except Exception:
            files_locator = page.locator(
                "a[href*='/files/'], a.download, .files-list a"
            )

        file_count = await files_locator.count()
        logger.info(f"[{appeal_num}] 🔍 Found {file_count} file link(s)")

        if file_count == 0:
            return True, order_num

        file_dir = os.path.join(output_dir, appeal_num)
        os.makedirs(file_dir, exist_ok=True)

        cookies = await extract_cookies_for_requests(context, auth_url)
        user_agent = await page.evaluate("() => navigator.userAgent")
        referer = appeal_url

        downloaded = 0
        for i in range(file_count):
            try:
                file_link = files_locator.nth(i)
                file_url = await file_link.get_attribute("href")
                file_name_raw = await file_link.text_content()
                file_name = sanitize_filename(
                    file_name_raw.strip() if file_name_raw else ""
                )
                file_url = normalize_url(file_url)
                if not (file_url and file_name):
                    continue
                file_path = os.path.join(file_dir, file_name)
                if os.path.exists(file_path) and is_valid_file(file_path):
                    downloaded += 1
                    continue
                if download_with_auth(
                    cookies, user_agent, referer, file_url, file_path
                ):
                    downloaded += 1
            except Exception as e:
                logger.error(f"[{appeal_num}] ✗ File #{i} error: {e}")

        logger.info(f"[{appeal_num}] 📦 Completed: {downloaded}/{file_count} files")
        return (downloaded > 0 or file_count == 0), order_num

    except Exception as e:
        logger.error(
            f"[{appeal_num}] ✗ Appeal error: {type(e).__name__}: {e}", exc_info=True
        )
        return False, None


async def process_batch_parallel(
    appeals: List[str],
    context: BrowserContext,
    base_appeal_url: str,
    output_dir: str,
    auth_url: str,
    max_concurrent: int = MAX_CONCURRENT,
) -> tuple[int, int, Dict[str, Optional[str]]]:
    """Обрабатывает батч обращений, возвращает (success, failed, {appeal_num: order_num})"""
    semaphore = asyncio.Semaphore(max_concurrent)
    order_map: Dict[str, Optional[str]] = {}

    async def process_with_semaphore(appeal_num: str):
        async with semaphore:
            page: Page = await context.new_page()
            try:
                success, order_num = await process_appeal_page(
                    page, context, appeal_num, base_appeal_url, output_dir, auth_url
                )
                if order_num:
                    order_map[appeal_num] = order_num
                return success
            finally:
                await page.close()

    tasks = [process_with_semaphore(appeal) for appeal in appeals]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    success = sum(1 for r in results if r is True)
    failed = len([r for r in results if r is False or isinstance(r, Exception)])
    if failed:
        logger.warning(f"📊 Batch completed: {success} OK, {failed} failed")
    return success, failed, order_map


async def main():
    # === Настройки URL ===
    initial_url = "https://csm.moscow.sportmaster.ru/call-center/v2/"
    target_url = "https://csm.moscow.sportmaster.ru/call-center/v2/entry"
    additional_url = "https://csm.moscow.sportmaster.ru/call-center/v2/appeals/1form"
    search_url = "https://csm.localcorp.net/"
    base_appeal_url = "https://csm.localcorp.net/appeal?app={}"  # ← исправлен шаблон
    auth_url = "https://csm.moscow.sportmaster.ru"

    base_folder = get_today_folder()
    output_dir = f"{base_folder}/appeals"
    logger.info(f"📁 Base folder: {base_folder}")

    async with async_playwright() as p:
        profile_path = os.path.abspath("data/browser")
        Path(profile_path).mkdir(parents=True, exist_ok=True)

        context: BrowserContext = await p.chromium.launch_persistent_context(
            user_data_dir=profile_path,
            channel="msedge",
            headless=False,
            args=[
                "--disable-gpu",
                "--disable-browser-side-navigation",
                "--no-first-run",
                "--no-default-browser-check",
            ],
            ignore_default_args=["--enable-automation"],
        )
        main_page: Page = (
            context.pages[0] if context.pages else await context.new_page()
        )

        try:
            # === Авторизация ===
            await main_page.goto(
                initial_url, wait_until="domcontentloaded", timeout=30000
            )
            await main_page.wait_for_url(f"*{target_url}", timeout=120000)
            logger.info(
                "✓ Auth successful"
                if target_url in main_page.url
                else f"⚠ Unexpected URL: {main_page.url}"
            )

            # === Токен + Поиск ===
            await main_page.goto(
                additional_url, wait_until="domcontentloaded", timeout=30000
            )
            await wait_for_page_ready(main_page)
            await main_page.goto(
                search_url, wait_until="domcontentloaded", timeout=30000
            )
            await wait_for_page_ready(main_page)

            logger.info(
                "=" * 80
                + "\n👤 USER ACTION: Set filters and click 'Найти'\n"
                + "=" * 80
            )
            await main_page.wait_for_selector(
                'a[href*="/appeals/1form/"]', timeout=300000
            )

            # === Парсинг записей ===
            records = await parse_results_table_with_pagination(main_page)
            if not records:
                logger.error("✗ No records parsed")
                sys.exit(1)

            appeal_numbers = [r["appeal_num"] for r in records if r.get("appeal_num")]
            logger.info(f"✓ Ready to process {len(appeal_numbers)} appeals")

            # === Обработка обращений + сбор order_num ===
            all_order_nums: Dict[str, Optional[str]] = {}
            total = len(appeal_numbers)
            for batch_start in range(0, total, BATCH_SIZE):
                batch = appeal_numbers[batch_start : batch_start + BATCH_SIZE]
                batch_num = batch_start // BATCH_SIZE + 1
                total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
                logger.info(
                    f"📦 Batch {batch_num}/{total_batches} ({len(batch)} appeals)"
                )

                success, failed, order_map = await process_batch_parallel(
                    batch,
                    context,
                    base_appeal_url,
                    output_dir,
                    auth_url,
                    max_concurrent=MAX_CONCURRENT,
                )
                all_order_nums.update(order_map)

                if batch_start + BATCH_SIZE < total:
                    logger.info(f"😴 Pausing {BATCH_PAUSE}s...")
                    await asyncio.sleep(BATCH_PAUSE)

            # === Обновляем records order_num перед сохранением ===
            for record in records:
                appeal_num = record.get("appeal_num")
                if appeal_num and appeal_num in all_order_nums:
                    record["order_num"] = all_order_nums[appeal_num]

            excel_path = save_to_excel(records, base_folder)

            logger.info("=" * 80 + "\n🎉 COMPLETE!")
            logger.info(f"📊 Total records: {len(records)}")
            logger.info(
                f"🧾 Orders found: {sum(1 for r in records if r.get('order_num'))}"
            )
            logger.info(f"💾 Report: {excel_path}")
            logger.info(f"📁 Attachments: {output_dir}\n" + "=" * 80)

        except Exception as e:
            logger.critical(f"✗ Critical error: {type(e).__name__}: {e}", exc_info=True)
            try:
                error_shot = os.path.join(
                    base_folder,
                    f"error_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                )
                await main_page.screenshot(path=error_shot, full_page=True)
                logger.info(f"✓ Screenshot: {error_shot}")
            except Exception as ss_err:
                logger.error(f"✗ Screenshot failed: {ss_err}")
            sys.exit(1)
        finally:
            await context.close()
            logger.info("✓ Browser context closed.")


if __name__ == "__main__":
    asyncio.run(main())
