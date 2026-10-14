from groq import (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    APIStatusError
)


def get_groq_error_message(error: Exception) -> str:

    if isinstance(error, RateLimitError):
        retry_after = error.response.headers.get("retry-after")

        message = (
            "Elérted a Groq API jelenlegi használati korlátját. "
            "Kérlek, próbáld újra egy kicsit később."
        )

        if retry_after:
            message += f" Javasolt várakozási idő: {retry_after} másodperc."

        return message

    if isinstance(error, APITimeoutError):
        return (
            "A válaszgenerálás túl sokáig tartott. "
            "Kérlek, próbáld újra."
        )

    if isinstance(error, APIConnectionError):
        return (
            "Nem sikerült kapcsolódni a Groq szolgáltatásához. "
            "Ellenőrizd az internetkapcsolatot."
        )

    if isinstance(error, APIStatusError):
        if error.status_code == 413:
            return (
                "A kérdés és a beszélgetési előzmények túl hosszúak. "
                "Próbálj rövidebb kérdést feltenni vagy új beszélgetést kezdeni."
            )

        if error.status_code == 401:
            return "Groq API-hitelesítési hiba. Ellenőrizd az API-kulcsot."

        return (
            "A Groq szolgáltatásában átmeneti hiba történt. "
            "Kérlek, próbáld újra később."
        )

    return "Váratlan hiba történt a válaszgenerálás során."
