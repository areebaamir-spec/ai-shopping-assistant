from django.core.exceptions import ValidationError


class NumberRequiredValidator:
    """
    Enforces that a password contains at least one digit,
    per FR001's password requirement.
    """

    def validate(self, password, user=None):
        if not any(character.isdigit() for character in password):
            raise ValidationError(
                "Password must contain at least one number.",
                code="password_no_number",
            )

    def get_help_text(self):
        return "Your password must contain at least one number."