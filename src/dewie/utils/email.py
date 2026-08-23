# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

import smtplib
from email.mime.text import MIMEText

from dewie.config import settings


async def send_password_reset_email(to_email: str, reset_link: str) -> None:
    """Sends a password reset email using SMTP.

    If SMTP settings are not configured, this is a no-op.
    """
    if not settings.smtp_host:
        return

    msg = MIMEText(f"Please use the following link to reset your password: {reset_link}")
    msg["Subject"] = "Password Reset Request"
    msg["From"] = settings.smtp_from_email
    msg["To"] = to_email

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            if settings.smtp_user:
                server.starttls()
                server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(msg)
    except Exception as e:
        import logging
        logging.getLogger("dewie.api").error("Failed to send password reset email to %s: %s", to_email, e)
        raise e
