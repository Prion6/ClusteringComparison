
#Función recursiva (Se llama a si misma) para sacar las rutas
def get_paths(dictionary, current_path:str, paths:list): #Recive un diccionario, un string para ir actualizando la ruta y una lista para guardar todas las rutas
  if type(dictionary) is dict: #Si el objeto es un diccionario 
    for key in dictionary.keys(): #Por cada llave que tenga
      obj = dictionary[key] #Obtiene el objeto en esa llave
      get_paths(dictionary[key], current_path + key + '/', paths) #Y se llama a si misma, pero con el nuevo objeto y le agrega la llave a la ruta
  else: # si no es un diccionario
    for item in dictionary: # por cada cosa en el objeto que recivió (Se asume que es una lista por lo que es código vulnerable, hay que comprobar)
      current_path = current_path.rsplit('/', 2)[0] # Esto es un parche porque las rutas de splus están re mal hechas
      current_path += '/' + item + '/' # le agrega el item a la ruta (Se asume que la lista tiene strings asi que esto también es código vulnerable, hay que comprobar)
      paths.append(current_path) # guarda la ruta final en la lista de rutas

hips_cats = splusdata.get_hipscats() #obtiene las raices de las rutas de splus

paths = [] # lista vacía para guardar las rutas
get_paths(hips_cats, '', paths) # llama a la función recursiva

cleaned = list(dict.fromkeys(paths)) # Quita las rutas duplicadas, estas no son los links finales
#print(cleaned)


sources = [] # para guardar los links finales

for path in paths: # por cada ruta en la lista de rutas
  links = splusdata.get_hipscats(path) #obtiene todos los links
  for link in links: #por cada link
    source = link[0].rsplit('/', 1)[0] + '/' + path #para arreglar el buig de splus, los links vienen en tuplas y le borra los / 
    #esto se queda solo con el primer elemento de la tupla, le quita la parte final que es la parte de la ruta a la que le quito los /
    # y la reemplaza con el 'path' que eslo que debería ir
    sources.append(source) #lo agrega a la lista de sources

print(source) # los links a los catalogos quedan en sources