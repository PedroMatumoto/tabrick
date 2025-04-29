from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField(label='Selecione um arquivo CSV')
    question = forms.CharField(
        label='Pergunte sobre o arquivo',
        max_length=255,
        required=False,
        widget=forms.TextInput(attrs={'placeholder': 'Digite sua pergunta aqui...'})
    )
    context = forms.CharField(
        label='Forneça o contexto',
        max_length=255,
        required=False,
        widget=forms.TextInput(attrs={'placeholder': 'Digite o contexto aqui...'})
    )