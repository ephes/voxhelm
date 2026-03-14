from __future__ import annotations

import os

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Create or update the initial Voxhelm operator account."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--username",
            default=os.getenv("VOXHELM_BOOTSTRAP_OPERATOR_USERNAME", "jochen"),
        )
        parser.add_argument("--email", default=os.getenv("VOXHELM_BOOTSTRAP_OPERATOR_EMAIL", ""))
        parser.add_argument(
            "--password",
            default=os.getenv("VOXHELM_BOOTSTRAP_OPERATOR_PASSWORD", ""),
        )

    def handle(self, *args, **options) -> None:
        del args
        username = str(options["username"]).strip()
        email = str(options["email"]).strip()
        password = str(options["password"]).strip()
        if not username:
            raise CommandError("A username is required.")
        if not password:
            raise CommandError(
                "A password is required. Pass --password or set "
                "VOXHELM_BOOTSTRAP_OPERATOR_PASSWORD."
            )

        user_model = get_user_model()
        user, created = user_model.objects.get_or_create(
            username=username,
            defaults={"email": email, "is_staff": True, "is_active": True},
        )
        updated_fields: list[str] = []
        if email and user.email != email:
            user.email = email
            updated_fields.append("email")
        if not user.is_staff:
            user.is_staff = True
            updated_fields.append("is_staff")
        if not user.is_active:
            user.is_active = True
            updated_fields.append("is_active")
        user.set_password(password)
        updated_fields.append("password")
        user.save(update_fields=updated_fields)

        action = "Created" if created else "Updated"
        self.stdout.write(self.style.SUCCESS(f"{action} operator account: {username}"))
