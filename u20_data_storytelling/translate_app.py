from collections.abc import Iterable
from joblib import Memory

# import asyncio

# Imports the Google Cloud Translation library
from google.cloud import translate


cache_dir = "./zurich_cache_directory"
memory = Memory(cache_dir, verbose=0)


@memory.cache
def translate_list(
    list_of_strings: Iterable[str],
    project_id: str = "mrprime-349614",
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
    project_id: str = "mrprime-349614",
    source_lang: str = "de",
    target_lang: str = "en-US",
) -> dict[str, str]:
    """Translates a list of strings to a dictionary with the original text as key and the translated text as value."""
    response = translate_list(
        list_of_strings,
        project_id=project_id,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    # print(response)
    return translation_response_to_dict(list_of_strings, response)


# translate_text('Afghanischer Windhund', 'mrprimetranslator')
translate_list_to_dict(["Schäferhund", "Dackel", "Rottweiler", "Afghanischer Windhund"])
