import cv2
import numpy as np


# Configuracion de la camara y filtro para evitar detectar ruido pequeno.
CAMARA = 0
ANCHO = 640
ALTO = 480
AREA_MINIMA = 1_200
UMBRAL_SOMBRA = 200


def crear_detector_fondo():
    # Crea el algoritmo que aprende como se ve el fondo de la escena.
    return cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=25, detectShadows=True
    )


def extraer_objetos(frame, detector, kernel):
    # Compara el cuadro actual con el fondo aprendido.
    mascara = detector.apply(frame)

    # Convierte la deteccion en blanco (objeto) y negro (fondo).
    _, mascara = cv2.threshold(mascara, UMBRAL_SOMBRA, 255, cv2.THRESH_BINARY)

    # Elimina puntos de ruido y une partes cercanas del mismo objeto.
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Busca cada grupo de pixeles blancos conectado; cada grupo puede ser un objeto.
    cantidad, _, estadisticas, centroides = cv2.connectedComponentsWithStats(
        mascara, connectivity=8
    )

    objetos = []
    for etiqueta in range(1, cantidad):  # La etiqueta 0 siempre representa el fondo.
        x, y, w, h, area = estadisticas[etiqueta]
        # Solo se conserva si tiene un tamano suficiente.
        if area >= AREA_MINIMA:
            cx, cy = centroides[etiqueta]
            objetos.append((x, y, w, h, area, int(cx), int(cy)))

    return mascara, objetos


def dibujar_resultado(frame, objetos):
    # Copia el cuadro para dibujar sin alterar la imagen original.
    salida = frame.copy()
    for numero, (x, y, w, h, area, cx, cy) in enumerate(objetos, start=1):
        # Caja verde, punto rojo central y texto con los datos del objeto.
        cv2.rectangle(salida, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(salida, (cx, cy), 4, (0, 0, 255), -1)
        texto = f"Objeto {numero} | area: {area} px"
        cv2.putText(salida, texto, (x, max(22, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(salida, f"Objetos detectados: {len(objetos)}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(salida, "q/ESC: salir | b: capturar fondo | r: reiniciar", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return salida


def main():
    # Abre la camara principal de la computadora.
    camara = cv2.VideoCapture(CAMARA)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)

    if not camara.isOpened():
        print("No se pudo abrir la camara. Verifica que no este en uso.")
        return

    detector = crear_detector_fondo()  # Se inicia el aprendizaje del fondo.
    # Matriz de 5 x 5 usada para limpiar la mascara.
    kernel = np.ones((5, 5), dtype=np.uint8)

    print("Aprendiendo el fondo. Mantenga la escena quieta unos segundos.")
    while True:
        valido, frame = camara.read()
        if not valido:
            print("No se pudo leer un cuadro de la camara.")
            break

        # Procesa cada cuadro: detecta objetos y dibuja el resultado.
        mascara, objetos = extraer_objetos(frame, detector, kernel)
        resultado = dibujar_resultado(frame, objetos)

        cv2.imshow("Deteccion nativa de objetos", resultado)
        cv2.imshow("Mascara de primer plano", mascara)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("r"):
            # Borra lo aprendido y vuelve a aprender el fondo.
            detector = crear_detector_fondo()
            print("Modelo de fondo reiniciado.")
        if tecla == ord("b"):
            # Actualiza el fondo con varios cuadros sin objetos enfrente.
            for _ in range(60):
                valido, fondo = camara.read()
                if valido:
                    detector.apply(fondo, learningRate=0.1)
            print("Fondo actualizado.")

    # Libera la camara y cierra las dos ventanas de OpenCV.
    camara.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    