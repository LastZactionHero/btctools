import requests

class GPT:
    def __init__(self, api_key):
        self.api_key = api_key

    def query(self, prompt):
        # Set up the API endpoint URL
        url = "https://api.openai.com/v1/chat/completions"

        # Send a POST request to the API
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                # 'prompt': prompt,
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
                # 'max_tokens': 50,  # Adjust the maximum number of tokens in the response
                "temperature": 0.6,  # Adjust the temperature for response randomness
                # 'n': 1,  # Adjust the number of responses to return
                # 'stop': None,  # Optional stop sequence to end the generated response
            },
        )
        data = response.json()
        result = data["choices"][0]["message"]["content"]
        return result