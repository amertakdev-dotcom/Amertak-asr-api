from __future__ import annotations

import re


class KhmerInverseText:
    # -----------------------
    # Khmer Digit Words
    # -----------------------
    DIGIT_MAP = {
        "សូន្យ": 0,
        "មួយ": 1,
        "ម៉ា": 1,
        "ពី": 2,  # common ASR omission of trailing រ
        "ពីរ": 2,
        "បី": 3,
        "បួន": 4,
        "ប្រាំ": 5,
        "ប្រាំមួយ": 6,
        "ប្រាំពីរ": 7,
        "ប្រាំបី": 8,
        "ប្រាំបួន": 9,
        "ហុក": 6,
        "ម្ភៃ": 20,
    }

    # -----------------------
    # Khmer Units
    # -----------------------
    UNIT_MAP = {
        "ដប់": 10,
        "សិប": 10,
        "រយ": 100,
        "ពាន់": 1000,
        "ម៉ឺន": 10000,
        "សែន": 100000,
        "លាន": 1000000,
    }

    # Prefix tens
    # Example:
    #   សាមសិប = 30
    #   សែសិប = 40
    #   ហាសិប = 50
    TENS_PREFIX_MAP = {
        "សាម": 30,
        "សែ": 40,
        "ហា": 50,
        "ហុក": 60,
        "ចិត": 70,
        "ប៉ែត": 80,
        "កៅ": 90,
    }

    NUMBER_TOKENS = sorted(
        set(DIGIT_MAP) | set(TENS_PREFIX_MAP) | {"ដប់", "សិប"} | set(UNIT_MAP),
        key=len,
        reverse=True,
    )

    KHMER_CHAR_PATTERN = r"\u1780-\u17FF"

    def normalize_text(self, text: str) -> str:
        """
        Normalize mixed English + Khmer text.

        Important:
        - Keep English spaces.
        - Remove spaces only between Khmer characters.
        - Fix truncated dollar word: ដុល្លា -> ដុល្លារ
        """
        if text is None:
            return ""

        text = str(text).strip()

        # Normalize repeated whitespace to one space first.
        # Example: "How   to   pay" -> "How to pay"
        text = re.sub(r"\s+", " ", text)

        # Remove spaces only when both sides are Khmer characters.
        # Example:
        #   "ដើម្បី ជៀសវាង ការផាកពិន័យ"
        #   -> "ដើម្បីជៀសវាងការផាកពិន័យ"
        #
        # But keep:
        #   "How to pay bills ដើម្បី"
        #   -> "How to pay bills ដើម្បី"
        text = re.sub(
            rf"(?<=[{self.KHMER_CHAR_PATTERN}])\s+(?=[{self.KHMER_CHAR_PATTERN}])",
            "",
            text,
        )

        # Only fix the truncated dollar form.
        # Avoid changing already-correct "ដុល្លារ" into "ដុល្លាររ".
        text = re.sub(r"ដុល្លា(?!រ)", "ដុល្លារ", text)

        return text

    def extract_number_phrase(
        self,
        clean_text: str,
        end_idx: int,
        limit_start: int = 0,
    ) -> tuple[str, int]:
        """
        Walk backwards from end_idx and grab the contiguous Khmer number words.
        Returns:
            phrase, start_index_of_phrase
        """
        tokens = []
        cursor = end_idx

        while cursor > limit_start:
            matched = False

            for token in self.NUMBER_TOKENS:
                start = cursor - len(token)

                if start < limit_start:
                    continue

                if clean_text[start:cursor] == token:
                    tokens.append(token)
                    cursor = start
                    matched = True
                    break

            if not matched:
                break

        tokens.reverse()
        phrase = "".join(tokens)

        return phrase, cursor

    def tokenize_number_words(self, text: str) -> list[str]:
        """
        Split a contiguous Khmer number phrase into known tokens.
        """
        tokens = []
        i = 0

        while i < len(text):
            matched = False

            for token in self.NUMBER_TOKENS:
                if text.startswith(token, i):
                    tokens.append(token)
                    i += len(token)
                    matched = True
                    break

            if not matched:
                return []

        # Merge patterns like "<digit>សិប" into tens prefix tokens if possible.
        # Example:
        #   បី + សិប -> សាម
        #   បួន + សិប -> សែ
        merged = []
        i = 0

        while i < len(tokens):
            if (
                tokens[i] in self.DIGIT_MAP
                and i + 1 < len(tokens)
                and tokens[i + 1] == "សិប"
            ):
                tens_val = self.DIGIT_MAP[tokens[i]] * 10

                prefix_token = None

                for key, value in self.TENS_PREFIX_MAP.items():
                    if value == tens_val:
                        prefix_token = key
                        break

                if prefix_token:
                    merged.append(prefix_token)
                else:
                    merged.append(tokens[i])
                    merged.append("សិប")

                i += 2
            else:
                merged.append(tokens[i])
                i += 1

        return merged

    def inverse_number_words(self, text: str) -> str:
        """
        Convert Khmer number words to digits.

        Examples:
            បី -> 3
            បីពាន់ -> 3000
            មួយសែន -> 100000
        """
        tokens = self.tokenize_number_words(text)

        if not tokens:
            return "0"

        total = 0
        current = 0
        i = 0

        while i < len(tokens):
            token = tokens[i]

            # Handle digits
            if token in self.DIGIT_MAP:
                current += self.DIGIT_MAP[token]
                i += 1
                continue

            # Handle tens prefixes
            # Example:
            #   ហា -> 50
            #   ហាសិប -> 50
            if token in self.TENS_PREFIX_MAP:
                current += self.TENS_PREFIX_MAP[token]

                if i + 1 < len(tokens) and tokens[i + 1] == "សិប":
                    i += 2
                else:
                    i += 1

                continue

            # Handle pure tens markers
            if token == "ដប់":
                current = 10 if current == 0 else current + 10
                i += 1
                continue

            if token == "សិប":
                current = 10 if current == 0 else current * 10
                i += 1
                continue

            # Handle large units
            if token in self.UNIT_MAP:
                unit = self.UNIT_MAP[token]

                if current == 0:
                    current = 1

                current *= unit
                i += 1

                if unit >= 1000:
                    total += current
                    current = 0

                continue

            i += 1

        return str(total + current)

    def parse_exchange_amount(self, text: str) -> dict:
        """
        Extract amount, currency, target_currency from normalized text.

        Rules:
        - If amount < 100 and currency is រៀល, auto-convert source currency
          to ដុល្លារ and target_currency to រៀល.
        - If only source currency is provided, target_currency defaults to the opposite.
        """
        clean = self.normalize_text(text)

        currency = None

        if "ដុល្លារ" in clean:
            currency = "ដុល្លារ"
        elif "រៀល" in clean:
            currency = "រៀល"

        if not currency:
            return {}

        idx = clean.index(currency)
        number_words, _ = self.extract_number_phrase(clean, idx)

        amount = self.inverse_number_words(number_words) if number_words else "0"

        try:
            amount_val = float(amount)
        except ValueError:
            amount_val = 0.0

        target_currency = None

        if currency == "ដុល្លារ" and "រៀល" in clean[idx + len(currency):]:
            target_currency = "រៀល"
        elif currency == "រៀល" and "ដុល្លារ" in clean[idx + len(currency):]:
            target_currency = "ដុល្លារ"

        if amount_val < 100 and currency == "រៀល":
            currency = "ដុល្លារ"
            target_currency = "រៀល"
        elif target_currency is None:
            target_currency = "ដុល្លារ" if currency == "រៀល" else "រៀល"

        return {
            "amount": int(amount_val) if amount_val.is_integer() else amount_val,
            "currency": currency,
            "target_currency": target_currency,
        }

    def convert(self, text: str) -> str:
        """
        Full sentence normalizer.

        Keeps English word spaces.
        Removes Khmer-to-Khmer spaces.
        Converts Khmer number words before currency to digits.
        """
        clean = self.normalize_text(text)

        # ---------------------------------------------------
        # CASE: <dollar><ដុល្លារ><cent><សេន>
        # Example:
        #   មួយដុល្លារហាសិបសេន -> 1.50ដុល្លារ
        # ---------------------------------------------------
        if "ដុល្លារ" in clean and "សេន" in clean:
            dollar_idx = clean.index("ដុល្លារ")
            cent_idx = clean.rindex("សេន")

            if cent_idx > dollar_idx:
                dollar_words, dollar_start = self.extract_number_phrase(
                    clean,
                    dollar_idx,
                )

                cent_words, _ = self.extract_number_phrase(
                    clean,
                    cent_idx,
                    dollar_idx + len("ដុល្លារ"),
                )

                if dollar_words and cent_words:
                    dollar_value = self.inverse_number_words(dollar_words)
                    cent_value = self.inverse_number_words(cent_words).zfill(2)

                    prefix = clean[:dollar_start]
                    after = clean[cent_idx + len("សេន"):]

                    return f"{prefix}{dollar_value}.{cent_value}ដុល្លារ{after}"

        # ---------------------------------------------------
        # CASE: ASR may output "រៀល" where user means dollar before cents
        # Example:
        #   មួយរៀលហាសិបសេន -> 1.50ដុល្លារ
        # ---------------------------------------------------
        if "រៀល" in clean and "សេន" in clean and "ដុល្លារ" not in clean:
            riel_idx = clean.index("រៀល")
            cent_idx = clean.rindex("សេន")

            if cent_idx > riel_idx:
                dollar_words, dollar_start = self.extract_number_phrase(
                    clean,
                    riel_idx,
                )

                cent_words, _ = self.extract_number_phrase(
                    clean,
                    cent_idx,
                    riel_idx + len("រៀល"),
                )

                if dollar_words and cent_words:
                    dollar_value = self.inverse_number_words(dollar_words)
                    cent_value = self.inverse_number_words(cent_words).zfill(2)

                    prefix = clean[:dollar_start]
                    after = clean[cent_idx + len("សេន"):]

                    return f"{prefix}{dollar_value}.{cent_value}ដុល្លារ{after}"

        # ---------------------------------------------------
        # CASE: cents only
        # Example:
        #   ហាសិបសេន -> 0.50ដុល្លារ
        # ---------------------------------------------------
        if "សេន" in clean and "ដុល្លារ" not in clean:
            cent_idx = clean.index("សេន")
            cent_words, cent_start = self.extract_number_phrase(clean, cent_idx)

            if cent_words:
                cent_value = self.inverse_number_words(cent_words).zfill(2)

                prefix = clean[:cent_start]
                after = clean[cent_idx + len("សេន"):]

                return f"{prefix}0.{cent_value}ដុល្លារ{after}"

        # ---------------------------------------------------
        # CASE: normal dollar
        # Example:
        #   បីដុល្លារ -> 3ដុល្លារ
        # ---------------------------------------------------
        if "ដុល្លារ" in clean:
            idx = clean.index("ដុល្លារ")
            number_words, start = self.extract_number_phrase(clean, idx)

            if number_words:
                number = self.inverse_number_words(number_words)

                prefix = clean[:start]
                after = clean[idx + len("ដុល្លារ"):]

                return f"{prefix}{number}ដុល្លារ{after}"

        # ---------------------------------------------------
        # CASE: normal riel
        # Example:
        #   បីពាន់រៀល -> 3000រៀល
        # ---------------------------------------------------
        if "រៀល" in clean:
            idx = clean.index("រៀល")
            number_words, start = self.extract_number_phrase(clean, idx)

            if number_words:
                number = self.inverse_number_words(number_words)

                prefix = clean[:start]
                after = clean[idx + len("រៀល"):]

                return f"{prefix}{number}រៀល{after}"

        return clean


# ------------------------------------------------
# Singleton instance
# ------------------------------------------------
_inverse_text_engine = KhmerInverseText()


# ------------------------------------------------
# FINAL FUNCTION
# app.py should import and use this function.
# ------------------------------------------------
def InverseText(text: str) -> str:
    return _inverse_text_engine.convert(text)


# ------------------------------------------------
# Optional compatibility helper functions
# ------------------------------------------------
def normalize_text(text: str) -> str:
    return _inverse_text_engine.normalize_text(text)


def tokenize_number_words(text: str) -> list[str]:
    return _inverse_text_engine.tokenize_number_words(text)


def inverse_number_words(text: str) -> str:
    return _inverse_text_engine.inverse_number_words(text)


def extract_number_phrase(
    clean_text: str,
    end_idx: int,
    limit_start: int = 0,
) -> tuple[str, int]:
    return _inverse_text_engine.extract_number_phrase(
        clean_text,
        end_idx,
        limit_start,
    )


def intent_exchange_rate_from_inverse_text(text: str) -> dict:
    return _inverse_text_engine.parse_exchange_amount(text)


# ------------------------------------------------
# Test examples
# ------------------------------------------------
if __name__ == "__main__":
    examples = [
        "How to pay bills on time ដើម្បី ជៀសវាង ការផាកពិន័យ បន្ថែម",
        "បី ដុល្លារ",
        "បី ពាន់ រៀល",
        "មួយ ដុល្លារ ហា សិប សេន",
        "មួយ រៀល ហា សិប សេន",
        "ហា សិប សេន",
        "I want to pay បី ពាន់ រៀល today",
    ]

    for example in examples:
        print(example, "=>", InverseText(example))