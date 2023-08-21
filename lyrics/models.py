from django.db import models

class GeneratedLyrics(models.Model):
    seed_text = models.CharField(max_length=200)
    next_words = models.IntegerField()
    generated_lyrics = models.TextField()

    def __str__(self):
        return self.generated_lyrics
