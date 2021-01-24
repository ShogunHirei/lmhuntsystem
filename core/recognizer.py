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
from PIL import Image
import pytesseract

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

def letters_clean(img):
    """
        File: recognizer.py
        Function Name: letters_clean
        Summary: Detecção Canny para os caracteres
        Description: Parametros utilizados para limpar a imagem real e destacar os caracteres

        img -> imagem sem manipulações
    """
    new_img = cv.Canny(img, 10, 55, apertureSize=3, L2gradient=True)
    return new_img
    

def  line_check(img, of_img):
    """
        File: recognizer.py
        Function Name: line_check
        Summary: Detectar linhas usando trasnformada Hough

        img -> manipulated image to identify lines
        of_img -> image to draw lines
    """
    lines = cv.HoughLinesP(img, 5, np.pi/90, 300, minLineLength=400, maxLineGap=1)

    rgb_img = cv.cvtColor(of_img, cv.COLOR_GRAY2BGR)

    if not lines is None:
        print('Linhas', len(lines))
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv.line(of_img,(x1,y1),(x2,y2),(255,255,0),2)
    else:
        print('Foi mal camarada!')

    return of_img, lines

    


def c_rects(img, img_res, *args, draw=True, limits=10, minArea=True):
    """
        File: recognizer.py
        Function Name: c_rects
        Summary: Find countours and draw Rectangles

        img -> Imagem com manipulações para encontrar os contornor
        img_res -> imagem na qual serão desenhados os retângulos
    """
    if not bool(args):
        args = [cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE,]

    contours, hier = cv.findContours(img, *args)

    if limits:
        greaterArea = sorted([ (idx, cv.contourArea(cont)) for idx,cont in enumerate(contours) ], 
                          key=lambda x: x[1], reverse=True)[1:10]
        contours = [ contours[i] for i, a in greaterArea]

    if draw:
        for c in contours:
            # find bounding box coordinates
            x,y,w,h = cv.boundingRect(c)
            cv.rectangle(img_res, (x,y), (x+w, y+h), (255, 255, 0), 2)

            if minArea:
                # find minimum area
                rect = cv.minAreaRect(c)
                # calculate coordinates of the minimum area rectangle
                box = cv.boxPoints(rect)
                # normalize coordinates to integers
                box = np.int0(box)
                # draw contours
                cv.drawContours(img_res, [box], 0, (255, 255, 0), 5)

    return contours, hier


# Para verificação do char usar opções 
# -c 80 20 -dl -40 0
# Necessário para posterior tratamento das condições do chat
def main(path, option, prcts, dy):
    # TODO: Desenhar na imagem: Retangulo de caixa de mensagens - FEITO!
    #                           Retângulo de cada mensagem - FEITO
    #   
    #       Remover desenhos e circundar verdadeira região de interesse (TROI)
    #           Sugestões:
    #                   -> Usar dilate e GaussianBlur e circundar região com desenho
    #                   -> Borrar região e extrair texto
    #                   -> usar máscara para desenhos em imagem manipulada
    #                   

    # Loading image array
    img_arr = cv.imread(path, 0)

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

    pimg= res_img.copy()

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
    # rgb_img = cv.cvtColor(img_arr, cv.COLOR_GRAY2BGR)

    # O segundo maior contorno, em área, será o que conterá provavelmente
    #   a caixa de mensagens, é possível usá-lo para extrair o fundo da imagem
    x,y,w,h = cv.boundingRect(contours[bigArea[0][0]])
    cv.rectangle(res_img, (x,y), (x+w, y+h), (0, 0, 55), 3)
    rect = cv.minAreaRect(contours[bigArea[0][0]])
    box = np.int0(cv.boxPoints(rect))
    # Desenhar o contorno agora é opcional
    # cv.drawContours(rgb_img, [box], 0, (85), 3)

    # pdb.set_trace()
    # Iniciando extração de zona fora do ROI
    ROI_y = box[:, 1]
    ROI_x = box[:, 0]

    y1, y2 = ROI_y.min(), ROI_y.max()
    x1, x2 = ROI_x.min(), ROI_x.max()

    img_arr = img_arr[y1:y2, x1:x2]
    res_img = res_img[y1:y2, x1:x2] 

    # Desenhar um retângulo ao redor de cada mensagem

    # Remover símbolos de carta e caixa de marcação com base em porcentagem
    #   remoção apenas na direção x
    nx1 = int( (x2-x1) * 0.165)
    nx2 = int( (x2-x1) * 0.995)
    res_img = res_img[:, nx1:nx2]
    img_arr = img_arr[:, nx1:nx2]

    # Atualizando imagem manipulada para novas manipulações
    res_img = img_arr.copy()

    
    # Borrar imagem para limitar 
    res_img = cv.GaussianBlur(res_img, (7,7), 0)
    res_img = cv.adaptiveThreshold(res_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 1)
    # Erode
    kernel2 = np.full((7,7), 1, dtype=np.uint8)
    res_img = cv.dilate(res_img, kernel, iterations=3)

    # res_img = cv.Sobel(res_img,cv.CV_64F,0,1,ksize=5)
    # res_img = np.uint8( np.absolute(res_img))

    # # Identificar linhas para limitar as mensagens
    img_arr, lines = line_check(~res_img.copy(), img_arr)

    # Linhas determinadas com sucesso
    # A função HougLinesP returna um array com os ponto da linha (x1,y1,x2,y2)  
    # Para isolar as mensagens usar a distancia entre y1 de cada linha e cortar a
    #   imagem original nesses pedaços

    dy = []
    p_y1 = 0

    # Filtrar apenas pontos y1
    lines = lines[:, 0, 1]
    # Organizar linhas em ordem decrescente
    lines.sort()

    # Inserir elemento nulo para manter integridade entre diferença de pontos
    lines = np.concatenate((np.zeros(1,), lines))
    # Obter diferença entre pontos para verificar se linhas não estão muito proximas
    lines = np.diff(lines)

    # Diferença de ponto ordenado permite filtragem para eliminar linhas sobrepostas
    cutPoints = [ int(pnt) for pnt in lines if pnt > 10]
    
    sp = len(cutPoints)
    old_pnt = 0
    msgs = []
    for idx, pnt in enumerate(cutPoints, 1):
        # Retirar estrela para facilitar para Tesseract
        temp_img = img_arr[old_pnt:pnt+old_pnt, :].copy()

        y, x = temp_img.shape
        y = int(y*0.325)
        x = int(x*0.8)

        temp_img[ y:, x: ] = 0
        msgs.append(temp_img)

        old_pnt = pnt

    # Mensagens filtradas!
    img_arr = msgs[-1]

    # Usando pytesseract contra a minha vontade
    print(*[pytesseract.image_to_string(msg) for msg in msgs])

    # Detectado texto em mensagens.

    # Passar para a análise de acordo com determinados usuários


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

