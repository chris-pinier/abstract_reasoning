"""Re-execute original plotting code, intercepting exports; preserve all original files."""
import os,ast,json,hashlib
from pathlib import Path
os.environ['MPLCONFIGDIR']='/tmp/elife-mpl-cache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pypdf import PdfReader,PdfWriter
BASE=Path(__file__).resolve().parent
DEST=BASE/'no_titles';DEST.mkdir(exist_ok=True)
originals={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*') if p.is_file() and p.name!='generate_no_titles.py' and not p.name.startswith('.')}
outputs=[]
def clean_save(fig,name,note):
 # Figure-level title/subtitle/footer artists, but not common axis labels.
 keep=[getattr(fig,'_supxlabel',None),getattr(fig,'_supylabel',None)]
 for text in list(fig.texts):
  if text not in keep:text.remove()
 # Asterisks referred to removed footnotes; retain full region names.
 for ax in fig.axes:
  yl=[t.get_text() for t in ax.get_yticklabels()]
  xl=[t.get_text() for t in ax.get_xticklabels()]
  if any(t.endswith(' *') for t in yl):ax.set_yticks(ax.get_yticks(),[t.removesuffix(' *') for t in yl])
  if any(t.endswith(' *') for t in xl):ax.set_xticks(ax.get_xticks(),[t.removesuffix(' *') for t in xl])
 if name=='03e_accuracy_profile_rank_clean':
  ax=fig.axes[0]
  labels=[t for t in ax.texts if t.get_position()[0]==-.065]
  names=[t for t in labels if not t.get_text().endswith('%')]
  percentages=[t for t in labels if t.get_text().endswith('%')]
  for label in names:
   value=min(percentages,key=lambda t:abs(t.get_position()[1]-label.get_position()[1]))
   from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
   model_text=TextArea(label.get_text(),textprops={'color':label.get_color(),'fontsize':10,'fontweight':'bold'})
   accuracy_text=TextArea(value.get_text(),textprops={'color':'black','fontsize':9,'fontweight':'normal'})
   packed=HPacker(children=[model_text,accuracy_text],align='baseline',pad=0,sep=8)
   ax.add_artist(AnnotationBbox(packed,(-.065,round(label.get_position()[1])),xycoords=ax.transData,
                               box_alignment=(1,.5),frameon=False,annotation_clip=False,pad=0))
   label.remove()
   value.remove()
  for text in ax.texts:
   if text.get_text().startswith('r = '):
    text.set_color('black');text.set_fontsize(9);text.set_fontweight('normal')
 fn=name+'_no_title'
 fig.savefig(DEST/(fn+'.pdf'),bbox_inches='tight',facecolor='white',dpi=600)
 fig.savefig(DEST/(fn+'.png'),bbox_inches='tight',facecolor='white',dpi=180)
 outputs.append({'file':fn,'source':name,'caption_kept_outside_figure':note})
 plt.close(fig)
class CleanExports(ast.NodeTransformer):
 def visit_FunctionDef(self,node):
  if node.name in ('save','finish'):
   # Both exporter signatures are fig, name, caption/note.
   args=[ast.Name(id=a.arg,ctx=ast.Load()) for a in node.args.args]
   node.body=[ast.Expr(value=ast.Call(func=ast.Name(id='clean_save',ctx=ast.Load()),args=args,keywords=[]))]
   return node
  return self.generic_visit(node)
 def visit_Expr(self,node):
  # Never overwrite the source data exports while recreating charts.
  call=node.value
  if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute):
   if call.func.attr in ('to_csv','write_text','mkdir'):return None
   if call.func.attr=='save' and isinstance(call.func.value,ast.Name) and call.func.value.id=='np':return None
  return self.generic_visit(node)
for script,cut in [('generate.py','# Audit and review booklet'),('generate_variants.py',"(OUT/'variants_notes.json')")]:
 src=(BASE/script).read_text().split(cut)[0]
 tree=ast.fix_missing_locations(CleanExports().visit(ast.parse(src)))
 namespace={'__file__':str(BASE/script),'__name__':'__clean_export__','clean_save':clean_save}
 exec(compile(tree,str(BASE/script),'exec'),namespace)
assert len(outputs)==14
writer=PdfWriter()
for item in sorted(outputs,key=lambda i:i['file']):
 p=DEST/(item['file']+'.pdf');reader=PdfReader(p);assert len(reader.pages)==1
 writer.append(reader,outline_item=item['source'])
writer.add_metadata({'/Title':'Figure variants without titles or notes'})
with (DEST/'00_all_figures_no_titles.pdf').open('wb') as f:writer.write(f)
(DEST/'captions_and_sources.json').write_text(json.dumps(outputs,indent=2))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in originals.items()),'Original file changed'
print(f'Created {len(outputs)} figure pairs plus a 14-page vector PDF collection in {DEST}. Originals unchanged.')
