import docx  
d = docx.Document(r'f:/Research/SGLab/cable_driven/Dexterous_Hand/code/synergy_hand_sim/docs/v0.1 结题_王裕硕_基于物理智能的绳驱折纸灵巧手顺序驱动技术研究.docx')  
with open('wang_out.txt','w',encoding='utf-8') as f:  
    f.write('WANG DOC\n')  
    for i,p in enumerate(d.paragraphs):  
        if p.text.strip():  
            f.write('[%d] %s\n' % (i, p.text.strip())) 
