from django import template

register = template.Library()

@register.filter
def get_item(obj, key):
    """
    Get an item from a dictionary or return None if not found.
    Also handles string inputs by returning them as is.
    """
    if isinstance(obj, dict):
        return obj.get(key)
    elif isinstance(obj, str):
        return obj
    return None 