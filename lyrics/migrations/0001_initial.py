# Generated by Django 4.1.5 on 2023-05-21 08:37

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='GeneratedLyrics',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('seed_text', models.CharField(max_length=200)),
                ('next_words', models.IntegerField()),
                ('generated_lyrics', models.TextField()),
            ],
        ),
    ]
