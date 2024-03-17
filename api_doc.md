## Overview

This document aims to introduce how to use our Text-to-Speech API, including making requests via GET and POST methods. This API supports converting text into the voice of specified characters and supports different languages and emotional expressions.

## Character and Emotion List

To obtain the supported characters and their corresponding emotions, please visit the following URL:

- URL: `http://127.0.0.1:5000/character_list`
- Returns: A JSON format list of characters and corresponding emotions
- Method: `GET`

```
{
    "Hanabi": [
        "default",
        "Normal",
        "Yandere",
    ],
    "Hutao": [
        "default"
    ]
}
```

## Regarding Aliases

From version 2.2.4, an alias system was added. Detailed allowed aliases can be found in `Inference/params_config.json`.

## Text-to-Speech

- URL: `http://127.0.0.1:5000/tts`
- Returns:  Audio on success. Error message on failure.
- Method: `GET`/`POST`

### GET Method

#### Format

```
http://127.0.0.1:5000/tts?character={{characterName}}&text={{text}}
```

- Parameter explanation:
  - `character`: The name of the character folder, pay attention to case sensitivity, full/half width, and language (Chinese/English).
  - `text`: The text to be converted, URL encoding is recommended.
  - Optional parameters include `text_language`, `format`, `top_k`, `top_p`, `batch_size`, `speed`, `temperature`, `emotion`, `save_temp`, and `stream`, detailed explanations are provided in the POST section below.
- From version 2.2.4, an alias system was added, with detailed allowed aliases found in `Inference/params_config.json`.

### POST Method

#### JSON Package Format

##### All Parameters

```
{
    "method": "POST",
    "body": {
        "character": "${chaName}",
        "emotion": "${Emotion}",
        "text": "${speakText}",
        "text_language": "${textLanguage}",
        "batch_size": ${batch_size},
        "speed": ${speed},
        "top_k": ${topK},
        "top_p": ${topP},
        "temperature": ${temperature},
        "stream": "${stream}",
        "format": "${Format}",
        "save_temp": "${saveTemp}"
    }
}
```

You can omit one or more items. From version 2.2.4, an alias system was introduced, detailed allowed aliases can be found in `Inference/params_config.json`.

##### Minimal Data:

```
{
    "method": "POST",
    "body": {
        "text": "${speakText}"
    }
}
```

##### Parameter Explanation

- **text**: The text to be converted, URL encoding is recommended.
- **character**: Character folder name, pay attention to case sensitivity, full/half width, and language.
- **emotion**: Character emotion, must be an actually supported emotion of the character, otherwise, the default emotion will be used.
- **text_language**: Text language (auto / zh / en / ja), default is multilingual mixed. 
- **top_k**, **top_p**, **temperature**: GPT model parameters, no need to modify if unfamiliar.

- **batch_size**: How many batches at a time, can be increased for faster processing if you have a powerful computer, integer, default is 1.
- **speed**: Speech speed, default is 1.0.
- **save_temp**: Whether to save temporary files, when true, the backend will save the generated audio, and subsequent identical requests will directly return that data, default is false.
- **stream**: Whether to stream, when true, audio will be returned sentence by sentence, default is false.
- **format**: Format, default is WAV, allows MP3/ WAV/ OGG.

