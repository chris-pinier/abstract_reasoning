"""Additional family-color variant; preserve the earlier versions."""
from pathlib import Path
import ast,json
BASE=Path(__file__).resolve().parent
p=BASE/'generate_no_titles.py'
ns={'__file__':str(p)}
exec(compile(p.read_text().split('for script,cut in ')[0],str(p),'exec'),ns)
palette=json.loads((BASE/'model_palette.json').read_text())
original=ns['clean_save']
def selected(fig,name,note):
 if name!='03e_accuracy_profile_rank_clean':
  ns['plt'].close(fig);return
 # Remove horizontal background guides from this variant.
 for line in list(fig.axes[0].lines):
  if line.get_color()=='#edf0f2':line.remove()
 # Reuse the established no-title/black-value formatting, but export a new name.
 original(fig,name,note)
# Redirect only the final target export names, preserving its styling rule.
import matplotlib.figure
old_save=matplotlib.figure.Figure.savefig
def save_variant(self,fname,*args,**kwargs):
 fname=Path(fname)
 if fname.stem=='03e_accuracy_profile_rank_clean_no_title':
  if not getattr(self,'_compact_family_layout',False):
   from matplotlib.text import Text
   ax=self.axes[0]
   pos=ax.get_position()
   ax.set_position([pos.x0,pos.y0,.32,pos.height])
   for text in self.findobj(match=Text):
    content=text.get_text()
    if content in ('Accuracy','Human-profile similarity'):text.set_fontsize(14)
    elif content.isdigit():text.set_fontsize(11)
    else:text.set_fontsize(13)
   # Fixed-width accuracy column: every percentage starts at the same x,
   # after an identical eight-point gap from the right-aligned model name.
   from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, DrawingArea
   for artist in list(ax.artists):
    if isinstance(artist,AnnotationBbox):
     children=artist.offsetbox.get_children()
     model=children[0].get_children()[0]
     value=children[1].get_children()[0]
     model_area=TextArea(model.get_text(),textprops={'fontsize':13,'fontweight':'bold','color':model.get_color()})
     value_area=DrawingArea(43,16,0,3)
     value_area.add_artist(Text(0,0,value.get_text(),fontsize=13,fontweight='normal',color='black',ha='left',va='baseline'))
     packed=HPacker(children=[model_area,value_area],align='baseline',pad=0,sep=8)
     ax.add_artist(AnnotationBbox(packed,artist.xy,xycoords=ax.transData,box_alignment=(1,.5),frameon=False,annotation_clip=False,pad=0))
     artist.remove()
   for collection in ax.collections:
    collection.set_sizes([260])
    collection.set_clip_on(False)
   self._compact_family_layout=True
  fname=fname.with_name('03f_accuracy_profile_rank_family_colors_no_title'+fname.suffix)
 return old_save(self,fname,*args,**kwargs)
matplotlib.figure.Figure.savefig=save_variant
src=(BASE/'generate_variants.py').read_text().split("(OUT/'variants_notes.json')")[0]
tree=ns['CleanExports']().visit(ast.parse(src))
# Set the family palette immediately after the original MC assignment.
for i,node in enumerate(tree.body):
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MC' for t in node.targets):
  tree.body.insert(i+1,ast.parse('MC = FAMILY_PALETTE.copy()').body[0]);break
else:raise RuntimeError('Missing model palette assignment')
tree=ast.fix_missing_locations(tree)
try:
 exec(compile(tree,str(BASE/'generate_variants.py'),'exec'),{'__file__':str(BASE/'generate_variants.py'),'clean_save':selected,'FAMILY_PALETTE':palette})
finally:matplotlib.figure.Figure.savefig=old_save
(BASE/'no_titles/03f_family_palette.json').write_text(json.dumps({'colors':palette,'note':'Gemma uses green shades; Llama and Llama-distilled DeepSeek use blue shades. Phi-4 and Qwen retain separate colors. Accuracy and similarity values remain black. All values and ranks unchanged.'},indent=2))
print('Created additional family-color PDF and PNG.')
