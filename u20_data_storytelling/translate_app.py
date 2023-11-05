from collections.abc import Iterable

# import asyncio

# Imports the Google Cloud Translation library
from google.cloud import translate


def translate_list(
    list_of_strings: Iterable[str],
    project_id: str = "mrprimetranslator",
    source_lang: str = "de",
    target_lang: str = "en-US",
) -> translate.TranslateTextResponse:
    """Translates a list, or another interable, of strings using Cloud Translation API.
    Returns a TranslateTextResponse object."""
    client = translate.TranslationServiceClient()
    location = "global"
    parent = f"projects/{project_id}/locations/{location}"
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": list_of_strings,
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": source_lang,
            "target_language_code": target_lang,
        }
    )
    return response


def translation_response_to_dict(
    list_of_strings: Iterable[str], response: translate.TranslateTextResponse
) -> dict:
    """Converts a TranslateTextResponse object to a dictionary
    with the original text as key and the translated text as value."""
    trans_dict = {}
    trans_dict = {
        text: translation.translated_text
        for (text, translation) in zip(list_of_strings, response.translations)
    }
    # print(trans_dict)
    return trans_dict


def translate_list_to_dict(
    list_of_strings: Iterable[str],
    project_id: str = "mrprimetranslator",
    source_lang: str = "de",
    target_lang: str = "en-US",
) -> dict[str, str]:
    """Translates a list of strings to a dictionary with the original text as key and the translated text as value."""
    response = translate_list(list_of_strings, project_id, source_lang, target_lang)
    # print(response)
    return translation_response_to_dict(list_of_strings, response)


# Initialize Translation client
# async def translate_text_async(
#     text: str = "YOUR_TEXT_TO_TRANSLATE", project_id: str = "YOUR_PROJECT_ID"
# ) -> translate.TranslationServiceClient:
#     """Translating Text."""

#     client = translate.TranslationServiceClient()

#     location = "global"

#     parent = f"projects/{project_id}/locations/{location}"

#     # Translate text from German to English
#     # Detail on supported types can be found here:
#     # https://cloud.google.com/translate/docs/supported-formats
#     response = client.translate_text(
#         request={
#             "parent": parent,
#             "contents": [text],
#             "mime_type": "text/plain",  # mime types: text/plain, text/html
#             "source_language_code": "de",
#             "target_language_code": "en-US",
#         }
#     )

#     # Display the translation for each input text provided
#     for translation in response.translations:
#         # print(f"Translated text: {translation.translated_text}")
#         return translation.translated_text


# async def translate_text_parallel(
#     text_list, project_id: str = "YOUR_PROJECT_ID"
# ) -> translate.TranslationServiceClient:
#     """Translates a list of texts in parallel using the Google Cloud Translation API asynchronously."""
#     # Create a list of tasks
#     tasks = [translate_text_async(text, project_id) for text in text_list]

#     # Execute the tasks concurrently
#     results = await asyncio.gather(*tasks)

#     # Return the translated texts
#     return results


# translate_text('Afghanischer Windhund', 'mrprimetranslator')
translate_list_to_dict(["Schäferhund", "Dackel", "Rottweiler", "Afghanischer Windhund"])
