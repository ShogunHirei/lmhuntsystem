"""
File: main.py
Author: ShogunHirei
Description: Arquivo principal da função para auxiliar na caçada de monstro
"""

import os
import json
import argparse


# Classe para lidar com gerenciamento de dados
class DataHandler:
    """
    Class:  DataHandler
        Classe para lidar com os arquivos em formato json 
        ser utilizada para lidar com as informações dos monstros

    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = self.decodingJson()
        self._pipe = 0
        
    
    def decodingJson(self):
        """
        Class:  DataHandler
        Method: decodingJson
        Description: Método para retornar lista de objetos com as 
                     informações em `filepath` 
        """
        decoder = json.JSONDecoder()
        try:
            # Reading data from filepath
            with open(self.filepath, 'r') as fp:
                data = decoder.decode(''.join(fp.readlines()))
        except:
            data = []

        return data


class MessageBox:
    """
    Class:  MessageBox
    Extrair caixa de mensagens do print e preparar para extração de texto e elementos

    """

    def __init__(self, path):
        self.path = path

    
    def mainbox(self):
        """
        Class:  MessageBox
        Method: mainbox
        Summary: extração da caixa principal de mensagem do centro
        """

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


