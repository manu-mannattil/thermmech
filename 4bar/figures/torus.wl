(* ::Package:: *)

(* ::Section:: *)
(*Shape space of the four-bar linkage*)


(* ::Text:: *)
(*The goal here is to plot the shape space of the four-bar linkage as curves on a torus.*)


(* ::Input:: *)
(*MyColorFunction[x_]:=Blend[{RGBColor["#88aacc"],RGBColor["#bbccdd"],RGBColor["#eeeeee"]},x];*)
(*Torus[R_,r_]:={ParametricPlot3D[{(R+r Cos[\[Theta]1])Cos[\[Theta]2],(R+r Cos[\[Theta]1])Sin[\[Theta]2],r Sin[\[Theta]1]},{\[Theta]1,0,2\[Pi]},{\[Theta]2,0,2\[Pi]},Mesh->None,ColorFunction->MyColorFunction,ColorFunctionScaling->True,PlotStyle->{Opacity[0.75]},Lighting->{"Ambient",White},PlotPoints->200]};*)
(*OnTorus[r_,R_,\[Theta]1_,\[Theta]2_]:={(R+r Cos[\[Theta]1])Cos[\[Theta]2],(R+r Cos[\[Theta]1])Sin[\[Theta]2],r Sin[\[Theta]1]};*)
(*Branches[R_,r_,\[Lambda]_]:={ParametricPlot3D[{(R+r Cos[\[Theta]1])Cos[\[Theta]1],(R+r Cos[\[Theta]1])Sin[\[Theta]1],r Sin[\[Theta]1]},{\[Theta]1,0,2\[Pi]},PlotStyle->RGBColor["#1f77b4"],PlotPoints->300],ParametricPlot3D[{(R+r Cos[\[Theta]1]) ((1+\[Lambda]^2)Cos[\[Theta]1]-2\[Lambda])/((1+\[Lambda]^2)-2\[Lambda] Cos[\[Theta]1]),(R+r Cos[\[Theta]1]) ((1-\[Lambda]^2)Sin[\[Theta]1])/(1+\[Lambda]^2-2\[Lambda] Cos[\[Theta]1]),r Sin[\[Theta]1]  },{\[Theta]1,0,2\[Pi]},PlotStyle->RGBColor["#d62728"],PlotPoints->300]};*)
(*MarkPoints[R_,r_,\[Lambda]_,\[Alpha]_,\[Beta]_]:={Graphics3D[{PointSize[0.015],Point[{(R+r Cos[\[Alpha]])Cos[\[Alpha]],(R+r Cos[\[Alpha]])Sin[\[Alpha]],r Sin[\[Alpha]]}]}],Graphics3D[{PointSize[0.015],Point[{(R+r Cos[\[Beta]]) ((1+\[Lambda]^2)Cos[\[Beta]]-2\[Lambda])/((1+\[Lambda]^2)-2\[Lambda] Cos[\[Beta]]),(R+r Cos[\[Beta]]) ((1-\[Lambda]^2)Sin[\[Beta]])/(1+\[Lambda]^2-2\[Lambda] Cos[\[Beta]]),r Sin[\[Beta]]}]}]};*)


(* ::Input:: *)
(*FourBar=Show[Torus[1,0.35]~Join~Branches[1,0.35,2],Axes->False,Boxed->False,ViewCenter->{0,0,0},ViewPoint->{15.`,6.`,10.`},ViewVertical->{0,0,1}]*)


(* ::Input:: *)
(*FourBarPoints=Show[Torus[1,0.35]~Join~Branches[1,0.35,2]~Join~MarkPoints[1,0.35,2,7\[Pi]/24,17\[Pi]/24],Axes->False,Boxed->False,ViewCenter->{0,0,0},ViewPoint->{15.`,6.`,10.`},ViewVertical->{0,0,1}]*)
(*Export["annotate.png",Rasterize[FourBarPoints,ImageResolution->300,Background->None]];*)


(* ::Input:: *)
(*Angles=ReadList["angles.txt", Real,RecordLists->True];*)


(* ::Input:: *)
(*Cloud[R_,r_]:={ListPointPlot3D[OnTorus[r, R,#1,#2]&@@@Angles,PlotStyle->{RGBColor["#8c9faf"],PointSize[0.007],Opacity[0.3]}]}*)


(* ::Input:: *)
(*FourBarCloud=Show[Torus[1,0.35]~Join~Cloud[1,0.35]~Join~Branches[1,0.35,2]~Join~MarkPoints[1,0.35,2,7\[Pi]/24,17\[Pi]/24],Axes->False,Boxed->False,ViewCenter->{0,0,0},ViewPoint->{15.`,6.`,10.`},ViewVertical->{0,0,1}]*)


(* ::Input:: *)
(*Export["4bar_cloud.png",Rasterize[FourBarCloud,ImageResolution->300,Background->None]];*)
