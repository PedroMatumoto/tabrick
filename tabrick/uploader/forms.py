from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField(label='Selecione um arquivo CSV')
    query = forms.CharField(
        label='Pergunte sobre o arquivo',
        max_length=255,
        required=False,
        widget=forms.TextInput(attrs={'placeholder': 'Digite sua pergunta aqui...'})
    )