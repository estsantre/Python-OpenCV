#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# La fuente de video es la cámara web
cap = cv2.VideoCapture(0)

kernelOp = np.ones((3,3),np.uint8)
kernel = np.ones((3,3),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX

# Reconocerá como un objeto a las secciones de un mismo color con un area mayor
# a areaTH

areaTH = 1000

# Analiza cada frame del video
while True:

    _, frame = cap.read()

    # Se definen los extremos de cada color, por ejemplo, el azul más claro y el
    # azul más oscuro. Todo azul entre estos extremos se tomará en cuenta

    # azules
    lower_blue = np.array([100,0,0])
    upper_blue= np.array([255,70,70])

    # rojos
    lower_red = np.array([0,0,70])
    upper_red= np.array([40,40,255])

    # verdes
    lower_green=np.array([0,80,0])
    upper_green=np.array([80,255,80])

    # amarillos
    lower_yellow = np.array([0, 190, 190])
    upper_yellow = np.array([180, 255, 255])

    # Se crean máscaras en donde todo será negro excepto el color de cada máscara
    maskBlue = cv2.inRange(frame, lower_blue,upper_blue)
    maskRed = cv2.inRange(frame, lower_red,upper_red)
    maskGreen = cv2.inRange(frame, lower_green,upper_green)
    maskYellow = cv2.inRange(frame, lower_yellow, upper_yellow)

    # Se eliminan los falsos positivos del fondo (MORPH_OPEN) y los falsos
    # negativos de las secciones (MORPH_CLOSE)

    maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_OPEN, kernel)
    maskBlue = cv2.morphologyEx(maskBlue, cv2.MORPH_CLOSE, kernel)

    maskRed = cv2.morphologyEx(maskRed, cv2.MORPH_OPEN, kernel)
    maskRed = cv2.morphologyEx(maskRed, cv2.MORPH_CLOSE, kernel)

    maskGreen = cv2.morphologyEx(maskGreen, cv2.MORPH_OPEN, kernel)
    maskGreen = cv2.morphologyEx(maskGreen, cv2.MORPH_CLOSE, kernel)

    maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_OPEN, kernel)
    maskYellow = cv2.morphologyEx(maskYellow, cv2.MORPH_CLOSE, kernel)

    # Por cada máscara buscará los contornos y cada contorno con un area mayor a
    # areaTH será un objeto. Luego, encerrará a cada objeto en un cuadro y pondrá
    # el centro y nombre de color.

    # Cada máscara se trabaja por separado, por lo que los cuadros se ponen todos
    # en el frame principal, en donde no hay máscara

    _, contours0, hierarchy = cv2.findContours(maskGreen,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            #cv2.drawContours(frame, cnt, -1, (0,255,0), 3, 8)
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.putText(frame,'Verde',(x,y), font, 1, (0,255,0), 1, cv2.LINE_AA)
            cv2.circle(frame,(cx,cy), 5, (0,0,255), -1)
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    _, contours0, hierarchy = cv2.findContours(maskRed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.putText(frame,'Rojo',(x,y), font, 1, (0,0,255), 1, cv2.LINE_AA)
            cv2.circle(frame,(cx,cy), 5, (0,0,255), -1)
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    _, contours0, hierarchy = cv2.findContours(maskBlue,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.putText(frame,'Azul',(x,y), font, 1, (255,0,0), 1, cv2.LINE_AA)
            cv2.circle(frame,(cx,cy), 5, (0,0,255), -1)
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    _, contours0, hierarchy = cv2.findContours(maskYellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(frame, 'Amarillo', (x, y), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Muestra el frame principal  

    cv2.imshow('Frame',frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
