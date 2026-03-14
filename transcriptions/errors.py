from __future__ import annotations


class ApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int = 400,
        error_type: str = "invalid_request_error",
    ):
        super().__init__(message)
        self.message = message
        self.status = status
        self.error_type = error_type
