from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, simpledialog
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import os

# Newspaper imports kept lazy to allow running without optional deps during setup
try:
    from newspaper import Article
    from newspaper.configuration import Configuration
except Exception:  # pragma: no cover
    Article = None
    Configuration = None

from cleaner import clean_corpus


CORPUS_FILE = "chat.txt"
CRAWL_INTERVAL = 3600  # seconds
MAX_LINKS_PER_BASE = 5


# --- Global state ---
gui = None  # will be set after Tk init
trainer_lock = threading.Lock()


# --- Initialize chatbot ---
chatbot = ChatBot("Chatpot")
trainer = ListTrainer(chatbot)


def safe_gui_status(message: str) -> None:
    """Thread-safe append to GUI status if available, else print."""
    global gui
    if gui is None:
        print(message)
        return
    try:
        gui.after(0, gui.append_status, message)
    except Exception:
        # Fallback to direct call if not in Tk thread yet
        gui.append_status(message)


# --- Train from local file if exists ---
if os.path.exists(CORPUS_FILE):
    try:
        cleaned_corpus = clean_corpus(CORPUS_FILE)
        with trainer_lock:
            trainer.train(cleaned_corpus)
        print("✅ Trained from local corpus.")
    except Exception as e:
        print(f"⚠️ Error training from local corpus: {e}")
else:
    print("⚠️ chat.txt not found, skipping local training.")


# --- URL list (initial) ---
base_urls = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://www.brainyquote.com/topics/motivational-quotes",
]


def resolve_language_from_url(url: str) -> str:
    if "wikipedia.org" in url:
        try:
            lang_code = url.split("//")[1].split(".")[0]
            return lang_code or "en"
        except Exception:
            return "en"
    return "en"


# --- Train from a single URL ---
def train_from_url(url: str) -> None:
    if Article is None or Configuration is None:
        safe_gui_status("⚠️ Newspaper3k not installed; skipping URL training.")
        return
    try:
        config = Configuration()
        config.request_timeout = 10
        config.browser_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/140.0.0.0 Safari/537.36"
        )

        language = resolve_language_from_url(url)

        article = Article(url, config=config, language=language)
        article.download()
        article.parse()
        text = article.text or ""
        if len(text) > 100:
            lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 20]
            if lines:
                with trainer_lock:
                    trainer.train(lines)
                safe_gui_status(f"✅ Trained from: {url}")
            else:
                safe_gui_status(f"⚠️ Extracted no substantial lines at: {url}")
        else:
            safe_gui_status(f"⚠️ Not enough text at: {url}")
    except Exception as e:
        safe_gui_status(f"⚠️ Error training from {url}: {e}")


# --- Crawl base URLs ---
def crawl_and_train(base_url: str) -> None:
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/140.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(base_url, headers=headers, timeout=10)
        if response.status_code != 200:
            safe_gui_status(f"⚠️ Failed to fetch {base_url}")
            return
        soup_links = set()
        soup = BeautifulSoup(response.text, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                soup_links.add(full_url)
            if len(soup_links) >= MAX_LINKS_PER_BASE:
                break
        # Train from base URL
        train_from_url(base_url)
        # Train from discovered links
        for link in soup_links:
            train_from_url(link)
    except Exception as e:
        safe_gui_status(f"⚠️ Error crawling {base_url}: {e}")


# --- Periodic background training ---
def periodic_training() -> None:
    while True:
        safe_gui_status("⏳ Starting periodic training...")
        for url in list(base_urls):
            crawl_and_train(url)
        safe_gui_status(f"✅ Training cycle completed. Waiting {CRAWL_INTERVAL} seconds...")
        time.sleep(CRAWL_INTERVAL)


# --- GUI Class ---
class ChatBotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chatpot 🌱")

        # Chat area
        self.chat_area = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=60, height=20)
        self.chat_area.pack(padx=10, pady=(10, 5))
        self.chat_area.config(state=tk.DISABLED)

        # Entry
        self.entry = tk.Entry(self, width=50)
        self.entry.pack(side=tk.LEFT, padx=(10, 0), pady=(0, 10))
        self.entry.bind("<Return>", self.send_message)

        self.send_btn = tk.Button(self, text="Send", command=self.send_message)
        self.send_btn.pack(side=tk.LEFT, padx=(5, 10), pady=(0, 10))

        # Status area
        self.status_area = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=60, height=8, fg="green")
        self.status_area.pack(padx=10, pady=(5, 10))
        self.status_area.config(state=tk.DISABLED)

        # Buttons for URL management
        self.add_url_btn = tk.Button(self, text="Add URL", command=self.add_url)
        self.add_url_btn.pack(side=tk.LEFT, padx=(10, 5), pady=(0, 10))
        self.remove_url_btn = tk.Button(self, text="Remove URL", command=self.remove_url)
        self.remove_url_btn.pack(side=tk.LEFT, padx=(5, 10), pady=(0, 10))

    # GUI utility to marshal to Tk thread
    def after(self, delay_ms: int, callback, *args):  # type: ignore[override]
        return super().after(delay_ms, callback, *args)

    def send_message(self, event=None):
        user_msg = self.entry.get().strip()
        if user_msg.lower() in (":q", "quit", "exit"):
            self.quit()
            return
        if user_msg:
            self._append_chat("You", user_msg)
            response = chatbot.get_response(user_msg)
            self._append_chat("Chatpot", str(response))
            self.entry.delete(0, tk.END)

    def _append_chat(self, sender, msg):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"{sender}: {msg}\n")
        self.chat_area.yview(tk.END)
        self.chat_area.config(state=tk.DISABLED)

    def append_status(self, msg):
        self.status_area.config(state=tk.NORMAL)
        self.status_area.insert(tk.END, f"{msg}\n")
        self.status_area.yview(tk.END)
        self.status_area.config(state=tk.DISABLED)

    def add_url(self):
        new_url = simpledialog.askstring("Add URL", "Enter the URL to train from:")
        if new_url:
            base_urls.append(new_url)
            self.append_status(f"Added URL: {new_url}")
            threading.Thread(target=crawl_and_train, args=(new_url,), daemon=True).start()

    def remove_url(self):
        if base_urls:
            removed_url = base_urls.pop()
            self.append_status(f"Removed URL: {removed_url}")
        else:
            self.append_status("No URLs to remove.")


# --- Run GUI and background training ---
if __name__ == "__main__":
    gui = ChatBotGUI()
    # Start periodic training in background
    training_thread = threading.Thread(target=periodic_training, daemon=True)
    training_thread.start()
    gui.mainloop()

