import datetime

from django import forms
from django.core.exceptions import ValidationError
from django.utils.translation import ugettext_lazy as _

class SearchForm(forms.Form):
    booking_url = forms.URLField()
