"""Formatting utilities for currency and numbers."""

def format_euro(amount):
    """Formats a number as Euro currency (e.g. 1.250 €)"""
    if amount is None:
        return "N/A"
    return f"{amount:,.0f} €".replace(",", ".")

def format_number(number):
    """Formats a number with thousands separators (e.g. 1.250)"""
    if number is None:
        return "N/A"
    return f"{number:,.0f}".replace(",", ".")
