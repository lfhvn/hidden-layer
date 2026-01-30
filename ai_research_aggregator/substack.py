"""
Substack API client.

Handles authentication and post creation/publishing via the Substack API.

Substack uses a REST API at https://{publication}.substack.com/api/v1/.
Authentication is done via session cookies (email + password login) or
API tokens. This client supports both approaches.

Usage:
    client = SubstackClient("your-publication")
    client.login("email@example.com", "password")

    # Or use a saved session token
    client = SubstackClient("your-publication", token="your_token")

    # Create a draft
    post_id = client.create_draft(
        title="AI Digest",
        subtitle="Today's highlights",
        body_html="<h1>Hello</h1>",
    )

    # Publish it
    client.publish(post_id)
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

TOKEN_PATH = os.path.expanduser("~/.config/ai-research-aggregator/substack_token")


@dataclass
class SubstackPost:
    """Represents a Substack post."""

    id: int
    title: str
    subtitle: str = ""
    slug: str = ""
    draft_url: str = ""
    published_url: str = ""
    is_published: bool = False


class SubstackClient:
    """Client for the Substack API."""

    def __init__(
        self,
        publication_url: str,
        token: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize the Substack client.

        Args:
            publication_url: Your Substack publication URL slug
                (e.g., "myresearch" for myresearch.substack.com).
            token: Session token for authentication.
            email: Email for login (if no token).
            password: Password for login (if no token).
        """
        # Normalize: strip protocol and .substack.com if provided
        pub = publication_url.strip()
        pub = pub.replace("https://", "").replace("http://", "")
        pub = pub.replace(".substack.com", "").rstrip("/")
        self.publication = pub

        self.base_url = f"https://{self.publication}.substack.com/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "HiddenLayerResearchAggregator/0.1",
            "Content-Type": "application/json",
        })

        self._user_id = None

        if token:
            self._set_token(token)
        elif email and password:
            self.login(email, password)
        else:
            # Try loading saved token
            saved = self._load_saved_token()
            if saved:
                self._set_token(saved)

    @property
    def is_authenticated(self) -> bool:
        """Check if we have authentication credentials."""
        return "substack.sid" in {c.name for c in self.session.cookies}

    def login(self, email: str, password: str) -> bool:
        """
        Log in to Substack with email and password.

        Returns True on success.
        """
        login_url = "https://substack.com/api/v1/login"
        payload = {
            "email": email,
            "password": password,
            "for_pub": self.publication,
        }

        response = self.session.post(login_url, json=payload, timeout=15)

        if response.status_code == 200:
            logger.info(f"Logged in to Substack as {email}")
            # Save token for future use
            for cookie in self.session.cookies:
                if cookie.name == "substack.sid":
                    self._save_token(cookie.value)
                    break
            return True

        logger.error(f"Substack login failed: {response.status_code} {response.text[:200]}")
        return False

    def create_draft(
        self,
        title: str,
        body_html: str,
        subtitle: str = "",
        section_id: Optional[int] = None,
    ) -> SubstackPost:
        """
        Create a new draft post on Substack.

        Args:
            title: Post title.
            body_html: HTML body content.
            subtitle: Post subtitle (appears below title in email).
            section_id: Optional section/category ID.

        Returns:
            SubstackPost with the draft details.
        """
        self._require_auth()

        payload: Dict[str, Any] = {
            "title": title,
            "subtitle": subtitle,
            "body_html": body_html,
            "type": "newsletter",
            "draft": True,
        }

        if section_id:
            payload["section_id"] = section_id

        response = self.session.post(
            f"{self.base_url}/drafts",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        post_id = data.get("id", 0)
        slug = data.get("slug", "")

        draft_url = f"https://{self.publication}.substack.com/publish/post/{post_id}"

        post = SubstackPost(
            id=post_id,
            title=title,
            subtitle=subtitle,
            slug=slug,
            draft_url=draft_url,
        )

        logger.info(f"Created draft: {title} (ID: {post_id})")
        return post

    def publish(
        self,
        post_id: int,
        send_email: bool = True,
    ) -> SubstackPost:
        """
        Publish a draft post.

        Args:
            post_id: The draft post ID to publish.
            send_email: Whether to send email to subscribers.

        Returns:
            SubstackPost with published details.
        """
        self._require_auth()

        payload = {
            "send": send_email,
        }

        response = self.session.put(
            f"{self.base_url}/drafts/{post_id}/publish",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        slug = data.get("slug", "")
        published_url = f"https://{self.publication}.substack.com/p/{slug}"

        post = SubstackPost(
            id=post_id,
            title=data.get("title", ""),
            subtitle=data.get("subtitle", ""),
            slug=slug,
            published_url=published_url,
            is_published=True,
        )

        logger.info(f"Published: {published_url}")
        return post

    def create_and_publish(
        self,
        title: str,
        body_html: str,
        subtitle: str = "",
        send_email: bool = True,
        section_id: Optional[int] = None,
    ) -> SubstackPost:
        """
        Create a draft and immediately publish it.

        Args:
            title: Post title.
            body_html: HTML body content.
            subtitle: Subtitle for email preview.
            send_email: Whether to send email to subscribers.
            section_id: Optional section/category ID.

        Returns:
            SubstackPost with published details.
        """
        draft = self.create_draft(
            title=title,
            body_html=body_html,
            subtitle=subtitle,
            section_id=section_id,
        )
        return self.publish(draft.id, send_email=send_email)

    def list_drafts(self) -> list:
        """List existing draft posts."""
        self._require_auth()

        response = self.session.get(
            f"{self.base_url}/drafts",
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    def _require_auth(self):
        """Raise if not authenticated."""
        if not self.is_authenticated:
            raise RuntimeError(
                "Not authenticated with Substack. Run:\n"
                "  ai-digest newsletter login\n"
                "Or set credentials in config."
            )

    def _set_token(self, token: str):
        """Set the session cookie from a token string."""
        self.session.cookies.set(
            "substack.sid",
            token,
            domain=".substack.com",
            path="/",
        )

    def _save_token(self, token: str):
        """Save token to disk for reuse."""
        os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
        with open(TOKEN_PATH, "w") as f:
            f.write(token)
        os.chmod(TOKEN_PATH, 0o600)
        logger.info(f"Token saved to {TOKEN_PATH}")

    @staticmethod
    def _load_saved_token() -> Optional[str]:
        """Load a previously saved token."""
        if os.path.exists(TOKEN_PATH):
            with open(TOKEN_PATH) as f:
                token = f.read().strip()
            if token:
                return token
        return None
