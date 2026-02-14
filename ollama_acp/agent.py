import asyncio
import base64
import logging
import secrets
from typing import Any

from acp import (
    PROTOCOL_VERSION,
    Agent,
    AuthenticateResponse,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    update_agent_message_text,
)
from acp.interfaces import Client
from acp.schema import (
    AgentCapabilities,
    ClientCapabilities,
    HttpMcpServer,
    Implementation,
    McpServerStdio,
    PromptCapabilities,
    SseMcpServer,
)

try:
    from ollama import AsyncClient as OllamaAsyncClient
except ImportError:
    raise ImportError("Please install ollama: pip install ollama")


class AgentSession:
    """Quản lý session cho mỗi cuộc hội thoại"""

    def __init__(self) -> None:
        self.pending_prompt: asyncio.Task[None] | None = None
        self.conversation_history: list[dict[str, Any]] = []


class OllamaAgent(Agent):
    """Agent Client Protocol adapter for Ollama"""
    _conn: Client

    def __init__(self, model: str = "llama3.2", ollama_host: str = "http://localhost:11434") -> None:
        """
        Khởi tạo OllamaAgent

        Args:
            model: Tên model Ollama (mặc định: llama3.2)
            ollama_host: URL của Ollama server (mặc định: http://localhost:11434)
        """
        self._sessions: dict[str, AgentSession] = {}
        self.model = model
        self.ollama_host = ollama_host

        # Tạo AsyncClient để gọi Ollama
        self.ollama_client = OllamaAsyncClient(host=ollama_host)

    async def list_models(self) -> list:
        """Liệt kê danh sách models có sẵn trong Ollama"""
        try:
            result = await self.ollama_client.list()
            # result.models là list của Model objects
            return result.models if hasattr(result, 'models') else []
        except Exception as e:
            logging.error(f"Failed to list models: {e}")
            return []

    async def verify_connection(self) -> bool:
        """Kiểm tra kết nối với Ollama server"""
        try:
            models = await self.list_models()

            # Lấy model names từ Model objects (attribute .model)
            model_names = []
            for m in models:
                if hasattr(m, 'model'):
                    model_names.append(m.model)
                elif isinstance(m, dict):
                    model_names.append(m.get('model', ''))

            # Kiểm tra xem model được chọn có tồn tại không
            model_exists = any(self.model in name or name.startswith(
                self.model) for name in model_names)

            if not model_exists:
                logging.warning(
                    f"Model '{self.model}' not found. Available models: {model_names}")
                logging.warning(f"Please run: ollama pull {self.model}")

            return True
        except Exception as e:
            logging.error(f"Failed to connect to Ollama: {e}")
            return False

    def on_connect(self, conn: Client) -> None:
        """Callback khi client kết nối"""
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        """Khởi tạo protocol handshake"""
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(
                load_session=False,
                prompt_capabilities=PromptCapabilities(image=True)
            ),
            agent_info=Implementation(
                name="ollama-agent",
                title="Ollama Agent",
                version="1.0.0"
            ),
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        """Xác thực - hiện tại không yêu cầu auth"""
        return AuthenticateResponse()

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer |
                          McpServerStdio] | None = None,
        **kwargs: Any
    ) -> NewSessionResponse:
        """Tạo session mới"""
        session_id = secrets.token_hex(16)
        self._sessions[session_id] = AgentSession()

        logging.info(f"Created new session: {session_id}")

        return NewSessionResponse(
            session_id=session_id,
            modes=None,  # Không sử dụng modes
        )

    async def prompt(
        self,
        prompt: list[Any],
        session_id: str,
        **kwargs: Any,
    ) -> PromptResponse:
        """Xử lý prompt từ user"""
        session = self._sessions.get(session_id)
        if not session:
            session = AgentSession()
            self._sessions[session_id] = session

        # Hủy prompt đang chạy nếu có
        if session.pending_prompt:
            session.pending_prompt.cancel()

        # Tạo task mới để xử lý prompt
        session.pending_prompt = asyncio.create_task(
            self.handle_prompt(session_id, prompt)
        )

        try:
            await session.pending_prompt
        except asyncio.CancelledError:
            return PromptResponse(stop_reason="cancelled")
        except Exception:
            logging.exception("Error handling prompt")
            raise
        finally:
            session.pending_prompt = None

        return PromptResponse(stop_reason="end_turn")

    async def handle_prompt(self, session_id: str, prompt: list[Any]) -> None:
        """Xử lý prompt với Ollama"""
        session = self._sessions[session_id]

        # Trích xuất text và images từ prompt
        prompt_text, images = self._extract_content_from_prompt(prompt)

        if not prompt_text and not images:
            return

        # Tạo message của user
        user_message = {
            "role": "user",
            "content": prompt_text
        }

        # Thêm images nếu có
        if images:
            user_message["images"] = images

        # Thêm message của user vào lịch sử
        session.conversation_history.append(user_message)

        try:
            # Gọi Ollama để sinh response (streaming)
            response_text = await self._stream_ollama_response(session_id, session)

            # Thêm response vào lịch sử
            session.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })

        except Exception as e:
            logging.exception("Error calling Ollama")
            await self._conn.session_update(
                session_id=session_id,
                update=update_agent_message_text(
                    f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
                ),
            )

    def _extract_content_from_prompt(self, prompt: list[Any]) -> tuple[str, list[bytes]]:
        """Trích xuất text và images từ prompt blocks"""
        prompt_text = ""
        images = []

        for block in prompt:
            # Handle text content
            if hasattr(block, "text"):
                prompt_text += block.text
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    prompt_text += block.get("text", "")
                elif block.get("type") == "image":
                    # Handle image content from dict
                    data = block.get("data", "")
                    if data:
                        try:
                            images.append(base64.b64decode(data))
                        except Exception as e:
                            logging.error(f"Failed to decode image: {e}")

            # Handle image content from object
            if hasattr(block, "type") and block.type == "image":
                data = getattr(block, "data", "")
                if data:
                    try:
                        images.append(base64.b64decode(data))
                    except Exception as e:
                        logging.error(f"Failed to decode image: {e}")

        return prompt_text.strip(), images

    async def _stream_ollama_response(self, session_id: str, session: AgentSession) -> str:
        """Stream response từ Ollama và cập nhật real-time"""
        full_response = ""

        try:
            # Gọi Ollama với stream=True để nhận response từng phần
            stream = await self.ollama_client.chat(
                model=self.model,
                messages=session.conversation_history,
                stream=True
            )

            # Stream từng chunk
            async for chunk in stream:
                if 'message' in chunk:
                    content = chunk['message'].get('content', '')
                    if content:
                        full_response += content

                        # Cập nhật message real-time
                        await self._conn.session_update(
                            session_id=session_id,
                            update=update_agent_message_text(content),
                        )

            return full_response

        except Exception as e:
            logging.exception("Error during Ollama streaming", e)
            raise

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """Hủy prompt đang chạy"""
        session = self._sessions.get(session_id)
        if session and session.pending_prompt:
            session.pending_prompt.cancel()

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Extension method handler"""
        return {"status": "ok"}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Extension notification handler"""
        pass


if __name__ == "__main__":
    # This is kept for backward compatibility when running agent.py directly
    import sys
    from .cli import main
    sys.exit(main())

