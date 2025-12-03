import zlib, os, uuid, json

def comprimir():
    name = str(uuid.uuid4()) + ".73nn"
    
    with open("ae118164-d092-11f0-9c2a-3010b3c1696f.73nn", "rb") as file:
        original_data = file.read()
    
    with open("modelo_comprimido.73nn", "wb") as file_out:
        file_out.write(zlib.compress(original_data, level=9))

def descomprimir():
    with open("modelo_comprimido.73nn", "rb") as file:
        decompressed_data = zlib.decompress(file.read())
        print(json.loads(decompressed_data))
        
comprimir()

descomprimir()