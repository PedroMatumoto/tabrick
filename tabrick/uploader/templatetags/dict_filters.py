from django import template
import markdown
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Gets an item from a dictionary."""
    return dictionary.get(key, None)

@register.filter
def markdown_to_html(text):
    """Converts markdown text to HTML."""
    if text is None:
        return ""
    return mark_safe(markdown.markdown(text))