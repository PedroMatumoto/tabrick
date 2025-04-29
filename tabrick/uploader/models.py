from django.db import models

class CSVUpload(models.Model):
    file = models.FileField(upload_to='csvs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
