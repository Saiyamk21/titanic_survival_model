import requests

response = requests.post(
    "https://saiyamkkkalls-tittanicspace.hf.space/run/predict",
    json={"data": [3, 22, 1, 0, 7.25, "male", "S"]}
)

print(response.status_code)
print(response.text)
