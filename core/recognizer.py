"""
File: recognizer.py
Author: ShogunHirei
Description: Arquivo para gerar função que reconhecerá texto e local
             de onde ele deverá ser extraído
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2 as cv
import pdb
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()


# Argumentos contendo informações sobre cortes de imagem
parser.add_argument('-c', '--cut-percs', nargs='+', type=float, default=[0,0],
                    help=''' Passe a porcentagem que deve ser cortada da imagem. 
                    Ex.: `-c 80,90 ` para cortar 80%% do eixo Y e 90%% do eixo X''')

parser.add_argument('-of', '--output-first', action='store_true', default=False,  
                    help=''' Se é para apresentar a imagem parcial (na qual está sendo trabalhada)
                            ou se é para mostrar a imagem original. Padrão é mostrar a imagem
                            alterada''')

# Argumentos contendo informações sobre cortes de imagem
parser.add_argument('-dl', '--delta', nargs='+', type=float, default=[0,0],
                    help=''' Deslocamento no eixo para mexer a imagem, (porcentagem)''')

# Identifucação da página
parser.add_argument('-n', '--number',  type=int, default=0,
                    help=''' Deslocamento no eixo para mexer a imagem, (porcentagem)''')



args = parser.parse_args()

PRCTGS = args.cut_percs
# Mostrar imagem verdadeira ou imagem manipulada
OPTS = args.output_first
DY = args.delta
NUM = args.number

# Relative file path
if NUM:
    num = NUM
else:
    num = np.random.randint(1,high=8)
FILEPATH = f'../data/raw_data/img{num}.jpeg'


# Para verificação do char usar opções 
# -c 80 20 -dl -40 0
# Necessário para posterior tratamento das condições do chat
def main(path, option, prcts, dy):
    # Loading image array
    img_arr = cv.imread(path, 0)
    # img_arr = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)

    # Image Manipulation
    kernel = np.ones((2,2), np.uint8)
    # Get Rectangle structure
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    # res_img = cv.morphologyEx(img_arr, cv.MORPH_GRADIENT, kernel, iterations=None)
    res_img = cv.Laplacian(img_arr, cv.CV_64F)
    res_img = np.uint8((res_img))

    # Canny Edge Detection para remover elementos desnecessários da imagem
    res_img = cv.Canny(img_arr, 50, 150, L2gradient=None)


    # Tentar capturar caixa central
    blur = cv.GaussianBlur(res_img, (7,7), 25, sigmaY=0)
    # ret, res_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # res_img = cv.adaptiveThreshold(res_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    res_img = cv.dilate(blur, kernel, iterations=7)

    # # Avaliando histograma das imagens
    # plt.hist(res_img.ravel(), 256)
    # plt.show()


    # # Identificação de linhas no dentro das mensagens
    # lines = cv.HoughLinesP(res_img, 5, np.pi/(180), 400, minLineLength=100,maxLineGap=10)
    # for line in lines:
        # x1, y1, x2, y2 = line[0]
        # # rho,theta = line[0]
        # # a = np.cos(theta)
        # # b = np.sin(theta)
        # # x0 = a*rho
        # # y0 = b*rho
        # # x1 = int(x0 + 1000*(-b))
        # # y1 = int(y0 + 1000*(a))
        # # x2 = int(x0 - 1000*(-b))
        # # y2 = int(y0 - 1000*(a))
        # cv.line(img_arr,(x1,y1),(x2,y2),(0,0,255),2)


    # # Função encontra satisfatoriamente todos os caracteres da imagem
    # #   mas também circunda elementos indesejados
    # ret, res_img = cv.threshold(res_img, 150, 255, 0)
    contours, hier = cv.findContours(res_img, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
    print('Contornos', len(contours))

    # Considerando que o maior contorno encontrado seja os limites da tela
    #   ele é eliminado para que seja possível separar a parte central 
    #   com as mensagens
    bigArea = sorted([ (idx, cv.contourArea(cont)) for idx,cont in enumerate(contours) ], 
                     key=lambda x: x[1], reverse=True)[1:10]

    print('Maior contorno', bigArea)

    # Convertendo imagem para RGB para fazer contorno colorido
    rgb_img = cv.cvtColor(img_arr, cv.COLOR_GRAY2BGR)

    # O segundo maior contorno, em área, será o que conterá provavelmente
    #   a caixa de mensagens, é possível usá-lo para extrair o fundo da imagem
    x,y,w,h = cv.boundingRect(contours[bigArea[0][0]])
    cv.rectangle(res_img, (x,y), (x+w, y+h), (0, 0, 55), 3)
    rect = cv.minAreaRect(contours[bigArea[0][0]])
    box = np.int0(cv.boxPoints(rect))
    # Desenhar o contorno agora é opcional
    cv.drawContours(rgb_img, [box], 0, (85), 3)

    # pdb.set_trace()
    # Iniciando extração de zona fora do ROI
    ROI_y = box[:, 1]
    ROI_x = box[:, 0]

    y1, y2 = ROI_y.min(), ROI_y.max()
    x1, x2 = ROI_x.min(), ROI_x.max()

    img_arr = img_arr[y1:y2, x1:x2]

    # for c in contours:
        # # find bounding box coordinates
        # x,y,w,h = cv.boundingRect(c)
        # cv.rectangle(res_img, (x,y), (x+w, y+h), (255, 255, 0), 2)
        # # find minimum area
        # rect = cv.minAreaRect(c)
        # # calculate coordinates of the minimum area rectangle
        # box = cv.boxPoints(rect)
        # # normalize coordinates to integers
        # box = np.int0(box)
        # # draw contours
        # cv.drawContours(img_arr, [box], 0, (255, 255, 0), 5)

    # TODO: Desenhar na imagem: Retangulo de caixa de mensagens - FEITO!
    #                           Retângulo de cada mensagem - FEITO
    #   
    #       Remover desenhos e circundar verdadeira região de interesse (TROI)
    #           Sugestões:
    #                   -> Usar dilate e GaussianBlur e circundar região com desenho
    #                   -> Borrar região e extrair texto
    #                   -> usar máscara para desenhos em imagem manipulada
    #                   




    # Image result show
    if option:
        cv.imshow(path, img_arr)
    else:
        cv.imshow(path, res_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return None


print(FILEPATH, OPTS, PRCTGS, DY, 'Before execution')
main(FILEPATH, OPTS, PRCTGS, DY)

