from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField(label='Selecione um arquivo CSV ou PDF')
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

class QueryForm(forms.Form):
    question = forms.CharField(
        label='Pergunta sobre CSV',
        max_length=255,
        required=False,
        widget=forms.TextInput(attrs={'placeholder': 'Digite sua pergunta aqui...'})
    )
    context = forms.CharField(
        label='Contexto',
        max_length=255,
        required=False,
        widget=forms.TextInput(attrs={'placeholder': 'Digite o contexto aqui...'})
    )
    pdf_query = forms.CharField(
        label='Consulta para PDF',
        max_length=1000,
        required=False,
        widget=forms.Textarea(attrs={'placeholder': 'Digite a consulta para o PDF...'})
    )
    selected_files = forms.MultipleChoiceField(
        label='Arquivos selecionados',
        required=False,
        widget=forms.CheckboxSelectMultiple,
        choices=[]
    )
    
    def __init__(self, *args, **kwargs):
        file_choices = kwargs.pop('file_choices', [])
        super(QueryForm, self).__init__(*args, **kwargs)
        self.fields['selected_files'].choices = file_choices