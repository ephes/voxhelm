from __future__ import annotations

import os

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import IntegrityError, transaction
from django.db.models import F

from operators.forms import LoginForm


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
        password = str(options["password"])
        if not username:
            raise CommandError("A username is required.")
        if not password:
            raise CommandError(
                "A password is required. Pass --password or set "
                "VOXHELM_BOOTSTRAP_OPERATOR_PASSWORD."
            )

        user_model = get_user_model()
        prepared = user_model(username=username)
        try:
            LoginForm.base_fields["username"].clean(username)
            user_model._meta.get_field("username").clean(username, prepared)
        except ValidationError as exc:
            raise CommandError("Invalid operator username: " + " ".join(exc.messages)) from exc
        if email:
            try:
                user_model._meta.get_field("email").clean(email, prepared)
            except ValidationError as exc:
                raise CommandError("Invalid operator email: " + " ".join(exc.messages)) from exc

        # Hash before taking the writer lock or publishing any account fields.
        prepared.set_password(password)
        try:
            with transaction.atomic():
                # A write obtains SQLite's writer lock before the read; on other
                # databases this also locks the existing row until commit.
                user_model.objects.filter(username=username).update(username=F("username"))
                user = user_model.objects.select_for_update().filter(username=username).first()
                created = user is None
                if user is None:
                    user = user_model(username=username)
                if email:
                    user.email = email
                user.is_staff = True
                user.is_active = True
                user.password = prepared.password
                if created:
                    user.save(force_insert=True)
                else:
                    user.save(update_fields=["email", "is_staff", "is_active", "password"])
        except IntegrityError as exc:
            raise CommandError("Account changed concurrently; retry bootstrap_operator.") from exc

        action = "Created" if created else "Updated"
        self.stdout.write(self.style.SUCCESS(f"{action} operator account: {username}"))
