# https://www.youtube.com/watch?v=vk4WWIreH8Q
# ^ A YouTube video I watched to gain the knowledge to encode/decode text files.

# This program encrypts a text document that will be used with the main.ipynb to
# show an example of allowing access to a file with a successful facial verification.

# The cryptography library will be used to encrypt, encode/decode the text file.
from cryptography.fernet import Fernet


# The function create key generates the access key for the encrypted file.
def create_key():
    # Fernet generates a key and saves it to access key.
    access_key = Fernet.generate_key()
    # The .key file is opened and has the access key written to it.
    with open('Facial_Recognition_Key.key', 'wb') as Facial_Recognition_Key:
        Facial_Recognition_Key.write(access_key)


# This function encrypt_fr_file encrypts the FR Encryption Example text file.
def encrypt_fr_file():
    # The .key file is read to gather the key for encrypting the text file.
    with open('Facial_Recognition_Key.key', 'rb') as key_file:
        fr_key = key_file.read()

    # The text file is opened and read to gather the data.
    with open('FR Encryption Example.txt', 'rb') as read_file:
        encrypt_data = read_file.read()

    # Fernet gathers the key and encrypts the data from the text file
    fernet = Fernet(fr_key)
    encrypted_document = fernet.encrypt(encrypt_data)

    # The encrypted data is written to the text file.
    with open('FR Encryption Example.txt', 'wb') as write_file:
        write_file.write(encrypted_document)


# This function decrypts the FR Encryption Example text file.
def decrypt_fr_file():
    # The key file is read to grab the key for decoding.
    with open('Facial_Recognition_Key.key', 'rb') as key_file:
        fr_key = key_file.read()

    # The text file is read to grab the encrypted data.
    with open('FR Encryption Example.txt', 'rb') as read_file:
        encrypted_data = read_file.read()

    # Fernet grabs the key and decrypts the encrypted text file
    fernet = Fernet(fr_key)
    decrypted_document = fernet.decrypt(encrypted_data)

    # The encrypted and decrypted text file is outputted.
    print('Encrypted file: ', encrypted_data.decode(), '\n')
    print('Decrypted file: ', decrypted_document.decode())


# Main just runs all the functions above.
# ***Note that if you encrypt the file, make sure to save the key somewhere before running
# the code again or else you will lose access to the file as it will now generate a
# different key than what is actually attached to the text file.***
def main():
    # Uncomment these functions below if using a new text file.
    # create_key()
    # encrypt_fr_file()
    decrypt_fr_file()


main()
