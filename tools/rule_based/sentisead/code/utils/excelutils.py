'''
Created on Sep 29, 2014

@author: gias
'''

import xlwt
import xlrd
import os

def open_workbook(name):
    if name.endswith('xls'):
        workbook = xlrd.open_workbook(name, formatting_info=True)
    else:
        workbook = xlrd.open_workbook(name+".xls", formatting_info=True)
    return workbook

def get_sheet_names(workbook):
    return workbook.sheet_names()

def get_worksheet(workbook_name, sheet_name):
    workbook = open_workbook(workbook_name)
    return workbook.sheet_by_name(sheet_name)

def get_sheet_rows(worksheet):
    num_rows = worksheet.nrows - 1
    curr_row = -1
    rows = list()
    while curr_row < num_rows:
        curr_row += 1
        row = worksheet.row(curr_row)
        rows.append(row)
    return rows

def get_sheet_contents(worksheet):
    num_rows = worksheet.nrows - 1
    num_cells = worksheet.ncols -1
    curr_row = -1
    headers = []
    contents = []
    while curr_row < num_rows:
        curr_row += 1
        curr_cell = -1
        row_content = []
        while curr_cell < num_cells:
            curr_cell += 1
            # Cell Types: 0=Empty, 1=Text, 2=Number, 3=Date, 4=Boolean, 5=Error, 6=Blank
            cell_type = worksheet.cell_type(curr_row, curr_cell)
            cell_value = worksheet.cell_value(curr_row, curr_cell)
            #print cell_type, cell_value,
            if curr_row == 0:
                headers.append(cell_value)
            else:
                #row_content.append(cell_value)
                if cell_type != 0 and cell_type != 6:
                    row_content.append(cell_value)
                else:
                    row_content.append(None)
        #print "\n"
        #print row_content
        contents.append(row_content)
    return headers, contents

def getSheetContents(fileNameWithPath, sheetName):
    worksheet = get_worksheet(fileNameWithPath, sheetName)
    headers, contents = get_sheet_contents(worksheet)
    return headers, contents

    

def style_font_title():
    font = xlwt.Font()
    font.name = 'Times New Roman'
    font.colour_index = 2
    font.bold = True
    #font.height = 1
    style = xlwt.Style.XFStyle()
    style.font = font
    
    return style

def style_font_post_body():
    font = xlwt.Font()
    font.name = 'Times New Roman'
    font.colour_index = xlwt.Style.colour_map['black']
    font.bold = True
    font.height = 10*20
    
    
    style = xlwt.Style.XFStyle()
    style.font = font
    alignment = xlwt.Alignment()
    alignment.wrap = 1
    alignment.horz = xlwt.Alignment.HORZ_LEFT
    alignment.shri = xlwt.Alignment.SHRINK_TO_FIT
    alignment.vert = xlwt.Alignment.VERT_TOP
    style.alignment = alignment
    
    borders = xlwt.Borders()
    #borders.bottom = xlwt.Borders.DASHED
    borders.right = xlwt.Borders.DASHED
    style.borders = borders
    return style

def style_font_post_sentence():
    style = xlwt.Style.XFStyle()
    #font = xlwt.Font()
    #font.name = 'Times New Roman'
    #font.colour_index = xlwt.Style.colour_map['black']
    #font.bold = True
    #font.height = 10*20
    #style.font = font

    alignment = xlwt.Alignment()
    alignment.wrap = 1
    #alignment.horz = xlwt.Alignment.HORZ_LEFT
    #alignment.shri = xlwt.Alignment.SHRINK_TO_FIT
    #alignment.vert = xlwt.Alignment.VERT_TOP
    style.alignment = alignment
    
    #borders = xlwt.Borders()
    #borders.bottom = xlwt.Borders.DASHED
    #borders.right = xlwt.Borders.DASHED
    #style.borders = borders
    #style = xlwt.easyxf("alignment: wrap on")
    return style

def style_hyperlink():
    font = xlwt.Font()
    font.name = 'Times New Roman'
    font.colour_index = xlwt.Style.colour_map['blue']
    style = xlwt.Style.XFStyle()
    style.font = font
    return style
    

def style_datetime():
    style = xlwt.Style.XFStyle()
    style.num_format_str = 'DD-MM-YYYY'
    return style


def get_hyperlink(url, label):
    
    link = 'HYPERLINK("' + url + '"; "'+label+'");'
    formula = xlwt.Formula(link)
    return formula

def getGenericStyleColWidths(headers):
    styles = []
    colWidths = []
    for header in headers:
        if header == 'sentence':
            styles.append(style_font_post_sentence())
            colWidths.append(256 * 4 * 10)
        elif header == 'body':
            styles.append(style_font_post_sentence())
            colWidths.append(256 * 10 * 10)
        else:
            styles.append(None)
            colWidths.append(-1)
    return styles, colWidths

class Spreadsheet(object):
    
    def __init__(self, dirSpreadsheets):
        self.dirSpreadsheets = dirSpreadsheets
    
    def setupThread(self, threadId, outfile=None):
        self.thread = threadId
        if outfile is None:
            outfile = str(self.thread)+".xls"
            self.outfile = os.path.join(self.dirSpreadsheets, str(self.thread)+'.xls')
        else:
            self.outfile = os.path.join(self.dirSpreadsheets, outfile)
        self.style_font_title = style_font_title()
        self.style_sentence = style_font_post_sentence()
        
        print self.outfile
        
        
    def set_labels(self, headers):
        self.labels = headers
    
    
    def set_styles(self, styles):
        self.styles = styles

    def set_col_widths(self, widths):
        self.colWidths = widths

    def guessAndSetStylesColWidths(self):
        styles, colWidths = getGenericStyleColWidths(self.labels)
        self.set_styles(styles)
        self.set_col_widths(colWidths)

    def get_title(self, ws):
        row_id = 0
        for i, label in enumerate(self.labels):
            ws.write(row_id, i, label, self.style_font_title)
        return ws

    def generate_excel(self, contents):
        wb = xlwt.Workbook()
        sheet_id = str("%s"%(str(self.thread)))
        ws = wb.add_sheet(sheet_id)
        ws.set_panes_frozen(True)
        ws.set_horz_split_pos(1) 
        ws = self.get_title(ws)
        
        for i, colWidth in enumerate(self.colWidths):
            if colWidth != -1:
                ws.col(i).width = colWidth

        row = 1
        for content in contents:
            #print content
            if len(content) == 0: continue
            for col, colVal in enumerate(content):
                colStyle = self.styles[col]
                if colStyle != None:
                    ws.write(row, col, colVal, style=colStyle)
                else:
                    
                    ws.write(row, col, colVal)
            row += 1
        wb.save(self.outfile)

class SpreadsheetGeneric(object):
    
    def __init__(self, dirSpreadsheets):
        self.dirSpreadsheets = dirSpreadsheets
    
    def guessAndSetStylesColWidths(self):
        styles, colWidths = getGenericStyleColWidths(self.labels)
        self.set_styles(styles)
        self.set_col_widths(colWidths)

    def setup(self, fileName, sheetName, headers, styles=None, colWidths=None): # without path
        
        if fileName.endswith('.xls') == False:
            fileName += ".xls"
        self.sheetName = sheetName 
        self.outfile = os.path.join(self.dirSpreadsheets, fileName)
        self.style_font_title = style_font_title()
        self.style_sentence = style_font_post_sentence()
        self.labels = headers
        if styles == None and colWidths == None:
            self.guessAndSetStylesColWidths()
        print self.outfile
        
    def set_styles(self, styles):
        self.styles = styles

    def set_col_widths(self, widths):
        self.colWidths = widths

    def __get_title(self, ws):
        row_id = 0
        for i, label in enumerate(self.labels):
            ws.write(row_id, i, label, self.style_font_title)
        return ws

    def generate_excel(self, contents):
        wb = xlwt.Workbook()
        sheet_id = self.sheetName
        ws = wb.add_sheet(sheet_id)
        ws.set_panes_frozen(True)
        ws.set_horz_split_pos(1) 
        ws = self.__get_title(ws)
        
        for i, colWidth in enumerate(self.colWidths):
            if colWidth != -1:
                ws.col(i).width = colWidth

        row = 1
        for content in contents:
            #print content
            if len(content) == 0: continue
            for col, colVal in enumerate(content):
                colStyle = self.styles[col]
                if colStyle != None:
                    ws.write(row, col, colVal, style=colStyle)
                else:
                    ws.write(row, col, colVal)
            row += 1
        wb.save(self.outfile)


class SpreadsheetOpinionSummary(object):
    
    def __init__(self, dirSpreadsheets):
        self.dirSpreadsheets = dirSpreadsheets
    
    def setupSheet(self, apiName):
        self.api = apiName
        outfile = str(self.api)+".xls"
        self.outfile = os.path.join(self.dirSpreadsheets, outfile)
        self.style_font_title = style_font_title()
        self.style_sentence = style_font_post_sentence()
        self.outfile = os.path.join(self.dirSpreadsheets, str(self.api)+'.xls')
        #print self.outfile
        
        
    def set_labels(self, headers):
        self.labels = headers
    
    def set_styles(self, styles):
        self.styles = styles

    def set_col_widths(self, widths):
        self.colWidths = widths

    def get_title(self, ws):
        row_id = 0
        for i, label in enumerate(self.labels):
            ws.write(row_id, i, label, self.style_font_title)
        return ws

    def generate_excel(self, contents):
        wb = xlwt.Workbook()
        sheet_id = "raw" #str("%s"%(str(self.api)))
        ws = wb.add_sheet(sheet_id)
        ws.set_panes_frozen(True)
        ws.set_horz_split_pos(1) 
        ws = self.get_title(ws)
        
        for i, colWidth in enumerate(self.colWidths):
            if colWidth != -1:
                ws.col(i).width = colWidth

        row = 1
        for content in contents:
            if len(content) == 0: continue
            for col, colVal in enumerate(content):
                colStyle = self.styles[col]
                if colStyle != None:
                    ws.write(row, col, colVal, style=colStyle)
                else:
                    ws.write(row, col, colVal)
            row += 1
        wb.save(self.outfile)
