import base64

with open('hadamard_base64.txt', 'r') as f:
    data = f.read().strip()

with open('Hadamard_Conjecture_Explainer_wofl.pdf', 'wb') as pdf_file:
    pdf_file.write(base64.b64decode(data))

print("Boom! Your PDF is ready, fren 😍")