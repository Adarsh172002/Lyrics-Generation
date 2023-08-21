from django import forms

class LyricsForm(forms.Form):
    seed_text = forms.CharField(label='Seed Text', max_length=100)
    next_words = forms.IntegerField(label='Next Words')
