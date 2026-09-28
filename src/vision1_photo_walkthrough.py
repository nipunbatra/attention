"""One continuous, measured image → patches → embeddings walkthrough."""
import base64
import json

ORDER = ['s01-rows', 'projection-size', 'real-patch-crops', 'patch-real-dimensions',
         'real-patch-normalize', 'patch-one-row-shape', 'patch-one-row-projection',
         'real-patch-projection', 'real-patch-shared', 'patch-projection-parameters',
         'real-patch-position', 'real-patch-qkv']


def add_photo_walkthrough(b, add):
    t,g,rect,arrow,line,image,mobile_rows = (b[k] for k in
        ['t','g','rect','arrow','line','image','mobile_rows'])
    data=json.loads((b['ASSETS']/'real-patch-path.json').read_text())
    uri='data:image/png;base64,'+base64.b64encode((b['ASSETS']/'model-input.png').read_bytes()).decode()
    evidence=' <a href="figures/vision1/real-patch-path.json">All 196 output vectors and selected patch inputs</a> · <a href="notebooks/vision/trace_real_patch.py">Reproduce the computation</a>.'
    selected=data['selected_patches']
    rgb=selected['63']['rgb_pixels']
    x=data['normalized_rgb_row']

    def num(v):return f'{v:.3f}'.replace('-','−')
    def vec(v,n=3,tail=True):return '['+', '.join(num(a) for a in v[:n])+(', …]' if tail else ']')
    def raw(v):return '['+', '.join(str(a) for a in v)+']'
    def box(x,y,w,label,color='c-e',size=28):
        return rect(x,y,w,64,color,'transparent')+t(x+w/2,y+42,label,size,color,'middle')
    def crop(x,y,side,number=63,outline=True):
        r,c=divmod(number-1,14)
        out=f'<svg x="{x}" y="{y}" width="{side}" height="{side}" viewBox="{c*16} {r*16} 16 16" overflow="hidden"><use href="#__PATCH_SOURCE__"/></svg>'
        if outline:out+=rect(x,y,side,side,'c-k' if number==64 else 'c-q','transparent',0)
        return out
    def grid(x,y,side):
        out=image(x,y,side,side,uri);cell=side/14
        for k in range(1,14):out+=line(x+k*cell,y,x+k*cell,y+side,'card',.7)+line(x,y+k*cell,x+side,y+k*cell,'card',.7)
        for n,color in [(63,'c-q'),(64,'c-k')]:
            r,c=divmod(n-1,14);out+=rect(x+c*cell,y+r*cell,cell,cell,color,'transparent',0)
        return out
    def mobile_crop(number=63):
        patch=crop(15,10,125,number).replace('<use href="#__PATCH_SOURCE__"/>',image(0,0,224,224,uri))
        return '<svg viewBox="0 0 360 155" role="img" aria-label="Actual patch '+str(number)+' from the dog image">'+patch+t(165,57,'P'+str(number),27,'c-k' if number==64 else 'c-q')+t(165,101,'16 × 16 × 3',23,'c-e')+'</svg>'
    def mobile_photo():
        return '<svg viewBox="0 0 360 350" role="img" aria-label="Dog image divided into 196 patches">'+grid(25,10,310)+t(180,347,'224 × 224 × 3',23,'c-e','middle')+'</svg>'
    def step(n,key,title,body,caption,question,point,prose,mobile):
        if '__PATCH_SOURCE__' in body:
            ref=key+'-photo-source'
            source=image(0,0,224,224,uri).replace('<image ',f'<image id="{ref}" ',1)
            body='<defs>'+source+'</defs>'+body.replace('__PATCH_SOURCE__',ref)
        add(key,f'{n} · {title}',body,caption,question,point,prose+evidence,mobile)

    # 1. Keep one image fixed for the whole worked example.
    body=image(35,35,335,335,uri)+t(202,418,'The same dog photograph',27,'c-e','middle')
    body+=t(475,68,'Input image',29,'ink-2')+t(475,137,'224 × 224 × 3',47,'c-e')
    body+=g(t(475,220,'224 rows of pixels',31)+t(475,275,'224 columns of pixels',31)+t(475,330,'3 values per pixel: R, G, B',31),1)
    body+=g(t(475,410,'Next: cut this image into equal-sized patches.',27,'c-e'),2)
    step(1,'s01-rows','Start with the same dog photograph',body,
         'We now carry this one image through the real model’s input pipeline. It has already been resized and cropped to 224 × 224 RGB pixels.',
         'What do the three image dimensions count?',
         'Keep the photograph fixed. Identify rows, columns and RGB channels before introducing any patch or embedding dimension.',
         'This is the exact prepared input to the saved ViT-Tiny checkpoint, before channel normalization. The original photograph is 500×334; evaluation preprocessing resizes and center-crops it to 224×224. '
         'The following twelve steps keep this same input. The earlier 4×4 grid was a coarse illustration; we now use the model’s actual 14×14 patch grid.',
         mobile_photo()+'<p>One prepared image: 224 rows × 224 columns × 3 RGB values.</p><p>Next, divide this image into patches.</p>')

    # 2. Put spatial directions and pixel extents on the photograph itself.
    def axes_photo(x,y,side,small=False):
        fs=19 if small else 23
        out=grid(x,y,side)
        out+=arrow(x,y-20,x+side+18,y-20,'ink-2')
        out+=arrow(x-22,y,x-22,y+side+18,'ink-2')
        for value,offset in [(0,0),(224,side)]:
            out+=line(x+offset,y-26,x+offset,y-14,'ink-2')
            out+=t(x+offset,y-34,str(value),fs,'ink-2','middle')
            out+=line(x-28,y+offset,x-16,y+offset,'ink-2')
            out+=t(x-34,y+offset+7,str(value),fs,'ink-2','end')
        out+=t(x+side/2,y-65,'x (pixels)',fs,'ink-2','middle')
        out+=f'<g transform="translate({x-(58 if small else 66)} {y+side/2}) rotate(-90)">'+t(0,0,'y (pixels)',fs,'ink-2','middle')+'</g>'
        return out

    body=axes_photo(100,95,280)
    body+=line(240,175,445,175,'c-q')+arrow(445,175,510,228,'c-q')+crop(520,140,176)+t(608,357,'P63 · enlarged',26,'c-q','middle')
    body+=t(790,132,'One patch',30,'c-e')
    body+=g(t(790,207,'16 pixels wide',31)+t(790,266,'16 pixels high',31)+t(790,325,'3 RGB channels',29),1)
    body+=t(500,421,'Next: number the pieces, one image row at a time.',26,'c-e')
    axes_mobile='<svg viewBox="0 0 360 360" role="img" aria-label="Dog image with x increasing to the right and y increasing down; both axes span 0 to 224 pixels">'+axes_photo(85,80,224,True)+'</svg>'
    step(2,'projection-size','Split the image into 16 × 16 patches',body,
         'The x-axis runs right; the y-axis runs down. Both span 224 pixels. Cut every 16 pixels along each axis. Each piece keeps its RGB values; we will count the pieces next.',
         'How many 16-pixel-wide pieces fit along each image axis?',
         'Point to the x-axis arrow, then the y-axis arrow and their endpoints. Follow the highlighted image region into the enlarged 16×16 patch. Leave the count for the next slide.',
         'The axes label image boundaries from 0 to 224 in pixel units, so their extent is 224 pixels. Pixel centers would instead be indexed 0 through 223. '
         'The y-axis points downward, matching image row order. Grid lines mark non-overlapping 16×16 cuts. '
         'The purple crop is P63, which we will locate in the row-by-row numbering next; the orange neighbor is P64. Counting patches has moved to the following slide.',
         axes_mobile+mobile_crop()+'<p>Cut every 16 pixels along x and along y. Count the resulting patches on the next slide.</p>')

    # 3. Show raster numbering using real crops; reveal the calculated endpoints.
    body=t(166,29,'column 1',25,'ink-2','middle')+t(306,29,'column 2',25,'ink-2','middle')+t(576,29,'last column',25,'ink-2','middle')
    for yy,row_name,numbers,reveal in [(62,'row 1',[1,2,14],0),(178,'row 2',[15,16,28],1),(325,'last row',[183,184,196],2)]:
        body+=t(30,yy+42,row_name,25,'ink-2')+t(445,yy+42,'…',35,'ink-2','middle')
        for xx,n in zip([134,274,544],numbers):
            body+=crop(xx,yy,64,n,False)+rect(xx,yy,64,64,'line','transparent',0)
            label=t(xx+32,yy+94,'P'+str(n),28,'c-e','middle')
            level=max(reveal,1 if n==14 else 0)
            body+=g(label,level) if level else label
    body+=t(166,310,'⋮',20,'ink-2','middle')+t(306,310,'⋮',20,'ink-2','middle')+t(576,310,'⋮',20,'ink-2','middle')
    body+=t(710,56,'Use the image axes:',29,'ink-2')
    body+=t(710,125,'columns: 224 ÷ 16',28)+t(710,190,'rows: 224 ÷ 16',28)
    body+=g(t(1030,125,'= 14',30,'c-e')+t(1030,190,'= 14',30,'c-e'),1)
    body+=t(710,270,'total = columns × rows',28)
    body+=g(t(710,334,'14 × 14 = 196 patches',31,'c-e')+t(710,407,'shape: 196 × 16 × 16 × 3',28,'c-e'),2)
    mobile_numbering='<svg viewBox="0 0 360 390" role="img" aria-label="Patches numbered left to right: P1, P2, through P14; P15, P16, through P28; last row P183, P184, through P196">'
    for yy,numbers in [(20,[1,2,14]),(148,[15,16,28]),(286,[183,184,196])]:
        for xx,n in zip([10,110,280],numbers):
            picture=crop(xx,yy,60,n,False).replace('<use href="#__PATCH_SOURCE__"/>',image(0,0,224,224,uri))
            mobile_numbering+=picture+rect(xx,yy,60,60,'line','transparent',0)+t(xx+30,yy+86,'P'+str(n),21,'c-e','middle')
        mobile_numbering+=t(224,yy+35,'…',30,'ink-2','middle')
    mobile_numbering+=t(40,276,'⋮',23,'ink-2','middle')+t(140,276,'⋮',23,'ink-2','middle')+t(310,276,'⋮',23,'ink-2','middle')+'</svg>'
    step(3,'real-patch-crops','Number the patches row by row',body,
         'Number left to right, then continue on the next row. Use the image and patch sizes to calculate the row length and total before revealing the answers.',
         'What is the last patch number in the first row? What starts the second row, and what is the final patch number?',
         'Start with P1 and P2. Ask students to divide each 224-pixel extent by 16, then reveal P14 and the second row. Multiply the two counts before revealing P196 and the full array shape.',
         'Each pictured tile is the actual patch with the stated index. Horizontal dots omit intermediate columns; vertical dots omit intermediate image rows. '
         'There are 224/16=14 patch columns and 14 patch rows. Row one is P1 through P14; row two is P15 through P28. The last row starts at 13×14+1=183 and ends at 14×14=196. '
         'In one-based indexing, patch number=(row−1)×14+column. Thus the previously highlighted P63 is row 5, column 7, and P64 is its right-hand neighbor. '
         'The resulting array has shape (196,16,16,3): patches, pixel rows, pixel columns, RGB channels.',
         mobile_numbering+mobile_rows(['Count','Calculation'],[['Columns','224 ÷ 16 = 14'],['Rows','224 ÷ 16 = 14'],['Total patches','14 × 14 = 196']])
         +'<p><strong>Patch array: 196 × 16 × 16 × 3.</strong></p><p>P63 is row 5, column 7; P64 is the next patch to its right.</p>')

    # 4. Read actual channel values, keeping the photograph visible.
    body=crop(35,75,256)+t(163,45,'P63 · enlarged',27,'c-q','middle')
    for i in range(1,16):body+=line(35+i*16,75,35+i*16,331,'card',.7)+line(35,75+i*16,291,75+i*16,'card',.7)
    body+=rect(35,75,16,16,'c-q','transparent',0)+rect(51,75,16,16,'c-k','transparent',0)
    body+=t(163,381,'16 × 16 = 256 pixels',27,'c-e','middle')
    for xx,label in [(405,'Pixel in P63'),(730,'R'),(865,'G'),(1000,'B')]:body+=t(xx,70,label,27,'ink-2')
    for j in range(2):
        yy=153+j*82;marks=t(405,yy,'first' if j==0 else 'second',29,'c-q' if j==0 else 'c-k')
        for xx,value,color in zip([730,865,1000],rgb[j],['c-a','c-d','c-e']):marks+=t(xx,yy,value,36,color)
        body+=marks if j==0 else g(marks,1)
    body+=g(t(405,329,'256 pixels × 3 RGB values',32,'c-e')+t(405,394,'= 768 numbers in this patch',34,'c-e'),2)
    step(4,'patch-real-dimensions','Read the RGB values inside patch 63',body,
         'P63 contains 256 pixels. Its first pixel is RGB [16, 17, 12]; its next pixel is [41, 42, 37]. Three values per pixel give 768 numbers in this same patch.',
         'How many input numbers does each pixel contribute?',
         'Use the two outlined pixels in the real crop. Read each RGB triple, then count 256 triples rather than introducing 768 without its source.',
         'These are the measured 8-bit pixel values from the prepared image. Start at the top-left pixel of P63 and move one pixel to the right for the second triple. '
         'The crop keeps shape (16,16,3). Its 256 RGB triples contain 768 scalar values. The grid drawn over the crop exposes its individual pixels.',
         mobile_crop()+mobile_rows(['Pixel','RGB'],[['First',raw(rgb[0])],['Second',raw(rgb[1])]])+'<p>16 × 16 = 256 pixels.<br>256 × 3 = 768 channel values.</p>')

    # 5. Explain the origin of the negative input numbers before showing them.
    body=crop(35,35,95)+t(160,70,'Same P63 pixels; apply the checkpoint’s normalization.',28)
    body+=t(35,161,'Pixel',26,'ink-2')+t(235,161,'8-bit RGB',26,'ink-2')+t(665,161,'Normalized RGB',26,'ink-2')
    for j in range(2):
        yy=226+j*82;marks=t(35,yy,str(j+1),30,'c-q')+t(235,yy,raw(rgb[j]),32)+arrow(495,yy-10,610,yy-10)+t(665,yy,vec(x[3*j:3*j+3],3,False),30,'c-e')
        body+=marks if j==0 else g(marks,1)
    body+=g(t(35,408,'Each channel: (value / 255 − 0.5) / 0.5',32,'c-e'),2)
    step(5,'real-patch-normalize','Normalize those same RGB values',body,
         'The checkpoint rescales every channel using the same formula. For the first red value, (16 / 255 − 0.5) / 0.5 ≈ −0.875. P63 still contains 768 values.',
         'Where does the first negative number come from?',
         'Carry the first RGB triple from the previous slide across the arrow. Work out the red channel before revealing the second pixel.',
         'The checkpoint uses channel mean 0.5 and standard deviation 0.5 after scaling 8-bit values by 255. This operation changes values but preserves the pixel arrangement and shape. '
         'Real preprocessing applies this formula to the image before patch extraction. Showing it on P63 gives the identical values because the operation acts independently on each channel. '
         'The earlier hand calculation used RGB/255 alone; here we use the pretrained checkpoint’s supplied normalization. All printed decimals are rounded.',
         mobile_crop()+mobile_rows(['Pixel 1','Values'],[['Raw RGB',raw(rgb[0])],['Normalized',vec(x[:3],3,False)]])+mobile_rows(['Pixel 2','Values'],[['Raw RGB',raw(rgb[1])],['Normalized',vec(x[3:6],3,False)]])
         +'<p>Apply (value / 255 − 0.5) / 0.5 to each channel. The patch still has 16 × 16 × 3 values.</p>')

    # 6. Flatten only after showing the patch and its actual scalar values.
    body=crop(35,100,192)+t(131,347,'P63',28,'c-q','middle')
    body+=arrow(35,77,227,77,'c-q')+line(227,77,251,77,'c-q')+line(251,77,251,89,'c-q')+line(251,89,23,89,'c-q')+line(23,89,23,112,'c-q')+arrow(23,112,35,112,'c-q')
    body+=t(285,50,'Read left to right; keep R, G, B together.',28)
    body+=t(365,125,'first pixel',25,'ink-2')+t(760,125,'second pixel',25,'ink-2')
    body+=g(t(320,205,'[',46,'c-e')+t(350,205,', '.join(num(v) for v in x[:3]),31,'c-e')
              +t(710,205,', '+', '.join(num(v) for v in x[3:6]),31,'c-e')+t(1070,205,', …]',35,'c-e'),1)
    body+=g(t(350,294,'x₆₃: 1 × 768',39,'c-e')+t(350,354,'1 patch row; 768 numbers in that row',28),2)
    body+=t(350,414,'Flattening rearranges values. It learns no weights.',25,'ink-2')
    step(6,'patch-one-row-shape','Flatten patch 63 into one row',body,
         'Copy the normalized RGB triples in pixel order: first pixel, second pixel, and so on. This gives x₆₃ with shape 1 × 768. The subscript 63 identifies the patch.',
         'Which three entries came from the second pixel?',
         'Follow the scan arrow across the patch, then point to the two corresponding triples in the row. Keep the same values visible across the transition.',
         'We use pixel-major RGB order: read three channels of one pixel, advance right, then continue on the next pixel row. '
         'The first axis in (1,768) counts patch rows; the second counts scalar features. There is one image throughout this walkthrough, and its batch axis is omitted. '
         'Flattening preserves all 768 normalized values and introduces no learned parameters.',
         mobile_crop()+mobile_rows(['Order in x₆₃','Normalized entries'],[['First pixel',vec(x[:3],3,False)],['Second pixel',vec(x[3:6],3,False)],['Continue','Remaining pixels in row order']])
         +'<p><strong>x₆₃: 1 × 768</strong><br>One patch row; 768 numbers in that row.</p>')

    # 7. Name and size the actual operation that consumes this row.
    body=crop(35,30,85)+t(145,78,'P63 is now x₆₃: a row of 768 normalized pixel values.',28)
    body+=box(35,172,260,'x₆₃ · 1 × 768',size=29)+arrow(295,204,390,204)
    body+=g(box(410,172,340,'nn.Linear(768, 192)',size=28)+arrow(750,204,845,204)+box(865,172,265,'c₆₃ · 1 × 192',size=29),1)
    body+=g(t(35,310,'768 inputs',31,'c-e')+t(430,302,'W_patch: 768 × 192',26,'c-e')+t(430,350,'bias: 192 values',26,'c-e')+t(865,310,'192 outputs',31,'c-e'),2)
    body+=t(35,425,'Each output = a weighted sum of the 768 inputs + its bias. No activation.',27)
    step(7,'patch-one-row-projection','Pass that row through the shared linear layer',body,
         'This layer reads 768 values and computes 192 output features. Its learned weights and biases are shared by every patch. One patch row enters; one embedding row leaves.',
         'How many weighted sums does this layer compute for P63?',
         'Carry x63 from the previous slide into the layer. Follow its row count and feature width separately, then identify the weight and bias shapes.',
         'The operation is c₆₃=x₆₃W_patch+b_patch. With row vectors, W_patch has shape (768,192); PyTorch stores the transposed Linear weight (192,768). '
         'There are 768×192=147,456 weights and 192 biases, totaling 147,648 shared parameters. D=192 is this model’s chosen embedding width. '
         'No ReLU or GELU follows this affine map. The checkpoint implements the equivalent operation with Conv2d(kernel_size=16,stride=16); the trace checks all patch rows against that implementation.',
         mobile_crop()+'<p>x₆₃ (1 × 768) → <strong>nn.Linear(768, 192)</strong> → c₆₃ (1 × 192).</p>'
         +mobile_rows(['Parameter','Shape'],[['W_patch','768 × 192'],['Bias','192 entries']])+'<p>Weighted sums + bias, with no activation. One shared set of 147,648 parameters.</p>')

    # 8. Show what came out; keep the same crop and operation in sight.
    body=crop(35,35,100)+arrow(145,85,230,85)+box(250,53,340,'nn.Linear(768, 192)',size=28)+arrow(590,85,685,85)+box(705,53,415,'c₆₃ · 192 output features',size=28)
    for xx,j in [(75,0),(335,1),(595,2),(975,191)]:
        marks=t(xx,222,'feature '+str(j+1),27,'ink-2')+t(xx,299,num(data['content'][j]),44,'c-e')
        body+=marks if j==0 else g(marks,1)
    body+=g(t(842,290,'…',46,'c-e'),1)+g(t(35,405,'c₆₃ = [−0.852, 1.339, 0.504, …, −1.199]      shape: 1 × 192',31,'c-e'),2)
    step(8,'real-patch-projection','Read the 192 output features for patch 63',body,
         'The layer produces c₆₃, the content embedding for P63. These are actual outputs from the pretrained model, rounded here. The first feature is −0.852 and the last is −1.199.',
         'Which patch do all 192 numbers describe?',
         'Trace the same crop through the same layer, then read individual output coordinates before collecting them into c63.',
         'Each displayed number is an output of the shared learned projection, computed from the exact dog-image input. '
         'The features are learned coordinates of a patch representation; the dimensions do not have manually assigned meanings. '
         'The output shape is (1,192). This is the real-scale version of collecting the two output neurons into c₁ in our earlier 12-to-2 calculation.',
         mobile_crop()+mobile_rows(['Output coordinate','Measured value'],[[str(j+1),num(data['content'][j])] for j in [0,1,2,191]])
         +'<p><strong>c₆₃: 1 × 192.</strong> One learned content representation of P63.</p>')

    # 9. Reuse exactly the same parameters on a second actual patch.
    body=t(35,35,'patch',25,'ink-2')+t(160,35,'input row · 1 × 768',25,'ink-2')+t(825,35,'output row · 1 × 192',25,'ink-2')
    for j,n in enumerate([63,64]):
        yy=80+j*188;v=selected[str(n)];marks=crop(35,yy,88,n)+t(79,yy+123,'P'+str(n),25,'c-q' if n==63 else 'c-k','middle')
        marks+=t(160,yy+45,vec(v['normalized_rgb_row'],2),25,'c-e')+arrow(393,yy+40,455,yy+40)
        marks+=arrow(745,yy+40,808,yy+40)+t(825,yy+45,vec(v['content'],2),26,'c-e')
        body+=marks if j==0 else g(marks,1)
    body+=rect(470,83,265,288,'c-e','transparent')+t(602,159,'same layer',29,'c-e','middle')+t(602,215,'768 → 192',32,'c-e','middle')+t(602,276,'same W and b',27,'c-e','middle')
    body+=g(t(35,429,'P63 and P64 produce different features using the same learned parameters.',27),2)
    step(9,'real-patch-shared','Pass patch 64 through the very same layer',body,
         'P64 is the patch immediately to the right of P63. Its different pixels give a different embedding. Both patches use the same weights and biases, and both produce 192 output features.',
         'Do we create new projection weights for P64?',
         'Keep P63’s path visible while revealing P64 underneath it. Point to the one shared layer and compare their measured input and output values.',
         'P64 is row 5, column 8 of the same image grid. Its first normalized RGB values are '+vec(selected['64']['normalized_rgb_row'])+'. '
         'Its first output coordinates are '+vec(selected['64']['content'])+'. Both were computed with the same checkpoint parameters as P63. '
         'Apply this operation independently to every patch. No patch-to-patch attention has happened at this stage.',
         mobile_crop()+mobile_rows(['P63','First two values'],[['Input',vec(x,2)],['Output',vec(data['content'],2)]])
         +mobile_crop(64)+mobile_rows(['P64','First two values'],[['Input',vec(selected['64']['normalized_rgb_row'],2)],['Output',vec(selected['64']['content'],2)]])
         +'<p>The same nn.Linear(768,192) layer produces both output rows.</p>')

    # 10. Carry row identity into the full output matrix.
    body=t(35,28,'One content embedding for every image patch',28,'c-e')
    for yy,n in [(50,1),(155,63),(235,64),(340,196)]:
        body+=crop(35,yy,62,n)+t(114,yy+41,'P'+str(n),26)+arrow(190,yy+30,245,yy+30)+t(275,yy+41,vec(selected[str(n)]['content']),27,'c-e')
    body+=t(388,145,'⋮',24,'ink-2')+t(388,331,'⋮',24,'ink-2')
    body+=box(740,52,380,'X: 196 × 768',size=33)+arrow(930,125,930,160)+box(740,173,380,'same Linear(768, 192)',size=27)
    body+=g(arrow(930,246,930,286)+box(740,299,380,'C: 196 × 192',size=33)+t(930,410,'196 patch rows; 192 features each',24,'c-e','middle'),1)
    table=mobile_rows(['Patch','First three output features'],[[str(i+1),vec(row)] for i,row in enumerate(data['all_content_rows'])])
    step(10,'patch-projection-parameters','Stack the 196 output rows into C',body,
         'Repeat the same projection for all 196 patches and keep their order. The input matrix X is 196 × 768. The output matrix C is 196 × 192: one content embedding per image patch.',
         'Which axis changes when we apply the shared projection to all patches?',
         'Match the illustrated crops to their measured output rows. Then carry the unchanged row count 196 from X through the layer to C.',
         'The displayed rows are P1, P63, P64 and P196; vertical dots mark omitted rows. Full 192-coordinate output vectors for all 196 patches are saved in the linked trace. '
         'The numerical check applies the same linear map to the complete X matrix and compares every output with the pretrained checkpoint’s patch embedding. '
         'C preserves patch order and has shape (196,192). Position information is the next operation.'
         +'<details><summary>Inspect all 196 output rows (first three coordinates)</summary>'+table+'</details>',
         mobile_rows(['Patch','Content embedding'],[[str(n),vec(selected[str(n)]['content'])] for n in [1,63,64,196]])
         +'<p>X (196 × 768) → shared layer → <strong>C (196 × 192)</strong>.</p><p>196 patch rows, each with 192 learned features.</p>')

    return ORDER
