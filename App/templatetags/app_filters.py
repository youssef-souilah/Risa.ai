from django import template
from django.template.defaultfilters import stringfilter

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """
    Get an item from a dictionary using the given key.
    Supports both string and integer keys.
    Returns None if the key doesn't exist.
    """
    if dictionary is None:
        return None
        
    try:
        # Try to convert key to integer if it's a string number
        if isinstance(key, str) and key.isdigit():
            key = int(key)
        return dictionary.get(key)
    except (AttributeError, TypeError, ValueError):
        return None 