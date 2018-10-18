Kedd Tutorialok:

Imitation Learning
https://sites.google.com/view/icml2018-imitation-learning/
Folyamatok megtanulására jó (szekvenciális döntéshozatal), expert segít tanulni példákkal, de problémás, ha olyan állapotba megy, amire nincs expert adat. 

Theoretical understanding of deep learning
http://unsupervised.cs.princeton.edu/deeplearningtutorial.html
Ez egy jó összefoglaló az egész témakörről, az n-dimenziós optimalizáció problémáiról a túl paraméterezés gyorsíthatja a konvergenciát, nagyon nem tudjuk elméletileg megmagyarázni a megfigyelt memória kapacitásokat. Zajstabilitás: a hálók stabilak a zajos inputra, de nem értjük, hogy ez miért van. A Generative  Adversarial hálóknál a diszkriminátornak elég összetettnek kell lennie, hogy értelems generátort tudjon betanítani.

Optimization Perspectives on Learning to Control
https://people.eecs.berkeley.edu/~brecht/l2c-icml2018
Folyamatvezérlés: (irányítás elmélet nagyon közel van Reinforcement Learninghez). Nézzük meg, hogy a lineáris esetet értjük-e, és az optimalizációs módszerek, és a megbízhatóság hogyan alakul egy lineáris problémánál, és ha ott értjük, akkor menjünk tovább a nemlineárisra. Jó áttekintése az optimalizációs módszereknek. A reinforcement learning is nehezen reprodukálható, sok trükk és bug teszi nehézzé ezt, ezért ellenőrizzük őket lineáris problémákon.

Szerda 1.
Predict and Constrain: Modeling Cardinality in Deep Structured Prediction
http://proceedings.mlr.press/v80/brukhim18a.html

Több cimkés klasszifikáció: kardinalitási probléma, hogyan lehet megkötni/explicitté tenni, hogy hány objektum van a képen, az adott cimkével (pl. 1 nap van a képen, v 3 macska). Dijsktra vetítési algoritmus, lecserélik a max függvényt -> softmax-ra, hogy deriválható legyen.
Igazából könnyebb a kardinalitást megmondani, mint a pontos cimkéket, ezért eléször becslik a kardinalitást, utána azt rögzítettnek véve optimalizálnak.
Amit kerülgetünk: nem tudunk optimalizálni nem differenciálható modelleket...



Improving Optimization in Models With Continuous Symmetry Breaking
http://proceedings.mlr.press/v80/bamler18a.html

A loss fv-nek van forgás szimmetriája, ezért lehet a részfiz módszereket matematikáját (lie algebrák, Goldstone bozonok, stb). A lényegesek a Goldstone bozonok lassítják a konvergenciát, ezért azokat le kell választani és külön lépésben optimalizálni. A Szimmetriákat arra használják h eliminálnak optimalizációs lépéseket, amelyeket így nem kell megcsinálni.




Conditional Neural Processes
http://proceedings.mlr.press/v80/garnelo18a.html

Nagyon kevés pontból nagyon stabil illesztést/közelítést ad (klasszifikációra és kép kiegészítésre is), nem zajos, de az előadásból nem értettük meg, hogy pontosan hogyan működik (Bayes + Gauss).




A Semantic Loss Function for Deep Learning with Symbolic Knowledge
http://proceedings.mlr.press/v80/xu18h.html

Logikai formulákat, kényszereket lehet kiróni a loss függvényre, ezeket valószínűségi alapon (Bernoulli formulák) próbálja teljesíteni, az optimalizáció végén általában nagyon nagy mértékben teljesülnek, és ezzel jobban tudja a háló a struktúrákat felismerni, jósolni.





Gradually Updated Neural Networks for Large-Scale Image Recognition
http://proceedings.mlr.press/v80/qiao18b.html

Ennek a fő észrevétele az, hogy nem feltétlenül szükséges nagyon mély hálókat gyártani, hasonló hatás érhető el azáltal is, ha az egy layeren belül több lépést csinálunk és a lépések a különböző csatornákat nem egyszerre léptetik, hanem a lépések között eltolva, néhányat korábban, néhányat később. Így kevesebb paraméterrel, kevesebb layerrel lehet ugyan azt a kifejezőképességet elérni.




Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks
http://proceedings.mlr.press/v80/xiao18a.html

Sok layer esetén probléma, hogy eltűnnek, felrobbannak a gradiensek (az optimalizáció rosszul kondícionált lesz) ezért nem lehet nagyon mély hálkat építeni. Itt azt csinálják, hogy átlagtér közelítésben tekintik a mély hálókat. Ennek segítségével olyan inicializációt írnak fel, amellyel a konv rétegek filterei ortogonálisra vannak inicializálva. Ekkor gyorsabban és jobban tanulnak. 

Csütörötökön volt ennek egy másik fele:

Dynamical Isometry and a Mean Field Theory of RNNs: Gating Enables Signal Propagation in Recurrent Neural Networks
http://proceedings.mlr.press/v80/chen18i.html
RNN-re mean field! Nem triviális állítás, főleg ha gating is van (memóriába becsatolás). Megjósolják a trainability-t.



The Dynamics of Learning: A Random Matrix Approach
http://proceedings.mlr.press/v80/liao18b.html
Lineáris hálón leírja az általánosítást, fitting, overfitting és egyéb problémák kezelését. ELőre meg tudják mondani a tanulás menetét (pl. hol kell leállítani a traininget).






SparseMAP: Differentiable Sparse Structured Inference
http://proceedings.mlr.press/v80/niculae18a.html

Ritka struktúra (gráf) reprezentáció + spec loss fv, amivel a gráfokat folytonos paraméterekkel lehet reprezentálni, kevés elem kombinációjaként és még deriváltható is. Inkább éleket tárolnak, mint a teljes struktúrát, mert azt könnyebb kezelni.




On the Optimization of Deep Networks: Implicit Acceleration by Overparameterization
http://proceedings.mlr.press/v80/arora18a.html
Ez a tutorial előadás egyik fejezetének kicsit bővebb kifejtése (matematikailag): a több paraméter segíti az optimalizációt.




Machine Theory of Mind
http://proceedings.mlr.press/v80/rabinowitz18a.html

Az emberi modell alkotást próbálják megérteni ágensek viselkedésének megfigyelésén keresztül. De mivel ez bonyolult, azt nézik h egy gép (háló) figyeli gépi ágensek (másik hálók) mozgását és meg kell jósolnia azt (belső state paramétereket/tulajdonságokat is). Ehhez egy jól ismert tesztet valósítanak meg, aminél az ágensek látóterén kívül megváltoztatják egy cél objektum korábban ismert helyét, ezért az ágensek rosszul hiszik h hol van az objektum, és a megfigyelő / jósló háló képes megtanulni, hogy mit is tud/gondol az ágens háló (rosszul) (a megfigyelő látja az ágens számára láthatatlan változtatást).

signSGD: Compressed Optimisation for Non-Convex Problems
http://proceedings.mlr.press/v80/bernstein18a.html
Ez egy elosztott SGD, ahol az adathalmazt több gépre osztják szét tanulni, és minden lépés után egy szerverre küldik a kiszámolt gradienseknek CSAK az előjelét és a szerver többségi szavazással döntei el h melyik paraméternek milyen előjelű legyen a változása. Ez jelentősen csökkenti a kommunikálandó információ mennyiséget, de belátható, hogy nem rontja el a konvergenciát.


NetGAN: Generating Graphs via Random Walks
http://proceedings.mlr.press/v80/bojchevski18a.html

Random gráfok generálása, úgy hogy a háló eloszlásokat tanul a gráfok felett. Ezt úgy érik el, hogy a hálókat random walk-okkal reprezentálják.


GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models
http://proceedings.mlr.press/v80/you18a.html

Ez egy olyan gráf generáló háló, amely nagyon jól meg tudja tanulni a gráfok sajátosságait anélkül h ezeket expliciten előírná az ember.





Programmatically Interpretable Reinforcement Learning
http://proceedings.mlr.press/v80/verma18a.html

Funkcionális domain specifikus nyelvet tanul a háló, így jobban tud általánosítani és ez később segít az interpretációban is (autós szimulációt tanul meg vezetni olyan esetekben is, amikor a tutorialban leírt expert advice nem létezik, mert ode nem merészkedett az expert soha (pl. pálya széle)).




Essentially No Barriers in Neural Network Energy Landscape
http://proceedings.mlr.press/v80/draxler18a.html

Valódi hálók loss fv szerkezetét nézték, és arra jutottak, hogy összefüggő a minimumok halmaza, tehát nincsenek benne nagy falak, kevéssé változik a loss fv a minimumok mentén, és el lehet jutni a tübbi lokális minimumba így könnyen.




Deep Linear Networks with Arbitrary Loss: All Local Minima Are Global
http://proceedings.mlr.press/v80/laurent18a.html

Amit a cím mond: lineáris hálókra (amikor az affin trafó után nincs nem lin) ezt be lehetett bizonyítani.



Csütörtök:

Learning unknown ODE models with Gaussian processes
http://proceedings.mlr.press/v80/heinonen18a.html

Csak egy köz diff rendszer állapotainak megfigyelésével elég jó jóslás tehető a rendszer későbbi állapotaira. A módszer csak a komponensek számában skálázik rosszul, a trajektóriák és megfigyelt pontok számában egyébként lineáris.


Differentiable Dynamic Programming for Structured Prediction and Attention
http://proceedings.mlr.press/v80/mensch18a.html

Ez nagyon érdekes állítás, de nem értettük meg, hogy működik azon kívül, hogy megint szétmossák a maximum függvényt smooth-max-ra.




Focused Hierarchical RNNs for Conditional Sequence Processing
http://proceedings.mlr.press/v80/ke18a.html

Ez a rendszer meg tudja tanulni, hogy hol keressen egy nagyobb adatban releváns blokkot, amin belül keresi a konkrét keresnivalót, tehát ilyen releváns kontextusokat tud felismerni és kezelni, így kevesebb adatot kell a hálónak kezelnie és jobban tanul.



Learning long term dependencies via Fourier recurrent units
http://proceedings.mlr.press/v80/zhang18h.html

Valami Fourier együtthatókat tanul a rendszer, amivel a hosszú távú korrelációk könnyebben tarthatóak "észben"


Investigating Human Priors for Playing Video Games
http://proceedings.mlr.press/v80/dubey18a.html

Egy egyszerű játék esetében hasonlították össze emberek és NN-ek játszását, és az a kérdés, miért tanul gyorsabban az ember. ELkezdték eliminálni a vizuális információkat, például a passzív elemek jelentését (létrák, platform, stb) ezek textúráját kicserélték h ne legyen felismerhető, és amikor már így az ember sem tudja intuitíven felismerni h melyik játék elem mire való, akkor ugyan olyan szintre kerül a hálóval. A háló teljesítményét nagyon rontja, ha az azonos jelentésű elemeket is elkezdik összekeverni (pl. más színűek).


Gradient Descent Learns One-hidden-layer CNN: Don't be Afraid of Spurious Local Minima
http://proceedings.mlr.press/v80/du18b.html

Ebben az egyszerű esetben meg lehetett mutatni, hogy a GD vagy egy lokális minimumba ragad be (25%) v a globálisba megy (75%), és ennek a fázisszerkezetét diszkutálták. (polinomiális konvergencia bizonyítható). Többször kell elindítani a tanítást ésakkor jó lesz:)


The Multilinear Structure of ReLU Networks
http://proceedings.mlr.press/v80/laurent18b.html

RELU hálózatok esetén a minimumoknak nincs második deriváltja (mert egész rész fv-ek vannak a lossban), ez általános jelenség, és ezek kezelésére mutatnak példát.


To Understand Deep Learning We Need to Understand Kernel Learning
http://proceedings.mlr.press/v80/belkin18a.html

Ebből önmagában nem értettük meg, h mik a kernel módszerek, de ezek, meg az interpolációs módszerek állítólag nagyon-nagyon hatékonyan megoldanak bizonyos problémákat (kb. egzaktul 0-hoz megy a költség fv!). Majd át kéne nézni a hivatkozásokat...


A Spline Theory of Deep Learning
http://proceedings.mlr.press/v80/balestriero18b.html

Ez is nagyon érdekesnek tűnt, de nem értettük az előadás alapján, hogy mit is nyertünk h ilyen max affin spline operátorokként lehet interpretálni a mély NN-eket....




Composable Planning with Attributes
http://proceedings.mlr.press/v80/zhang18k.html

Ez a rendszer attribútumokat tanul meg társítani pl. képekhez, és ezekből tud state-ket meg state átmeneteket felismerni és összekombinálni úgy, hogy egy adott célt elérjen. Ez is egy valamilyne értelemben redukált, hierarchikus feladat megoldási megközelítés.




Understanding the Loss Surface of Neural Networks for Binary Classification
http://proceedings.mlr.press/v80/liang18a.html

Bizonyos feltételek esetén bármelyik minimumot találjuk meg, az ugyan olyan jó, mint a többi. A feltételek nem teljesülését vizsgálták részletesen.



Geometry Score: A Method For Comparing Generative Adversarial Networks
http://proceedings.mlr.press/v80/khrulkov18a.html
Topologiai Homológiát használ, hogy összehasonlítsa az eredeti és a generált kép geometriai tulajdonságait és ez alapján hasonlíthatóak össze.


Péntek:

Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)
http://proceedings.mlr.press/v80/kim18d.html

Ez egy eszköz arra, hogy hogyan határozzuk meg, hogy milyen tulajdonságok alapján klasszifikált vlaamit a háló annak, aminek, úgy, hogy egy másik hálóval definiáljuk a tulajdonságot úgy, hogy azt traineljük a tulajdonsággal rendelkező és nem rendelkező adatokon, ebből kapunk egy vektort, és a másik, vizsgálni kívánt hálónk megfelelő rétegét lederiváljuk eeszerint a vektor szerint. Ez elég jónak és hasznosnak tűnt.


Learning equations for extrapolation and control
http://proceedings.mlr.press/v80/sahoo18a.html

Ez olyan réteget vesz, amiben mindenféle bonyolult függvények (pl. osztás, sin, cos) vna, és azokkal tanul meg összetett sok apraméteres formulákat illeszteni megfigyelt értékekre. Sokkal jobban jósol, mint más módszerek, és kinyerhető belőle a szimbolikus alak.



PDE-Net: Learning PDEs from Data
http://proceedings.mlr.press/v80/long18a.html

Ha jól értjük, a véges diff stencileket tanulja meg a háló, és "jól" szimulálja a diffegyenlet megoldását az időben előbbre haladva. Konvekciós és nem lineáris forrású diffúziós problémákat néztek csak.



Weightless: Lossy weight encoding for deep neural network compression
http://proceedings.mlr.press/v80/reagan18a.html

Csináltak egy még jobb tömörítő módszert mint az eddigiek, ami akár 1/500-ad részére képes hálókat tömöríteni. Jó összefoglalónak tűnt, hogy milyen trükkök léteznek etéren.




Building Machines that Learn and Think Like People
https://arxiv.org/abs/1604.00289

Miminden hiányzik ahhoz, hogy olyan AI-t csináljunk, mint az emberek?
Jobban megérteni a gyerekek pszichológiai fejlődését, több pszichológiai és párhuzamos AI kísérletet cisnálni, .pl a fizikai valóság megjóslására. Kiderül, hogy a geyrkek számos beépített információval és absztrakciós erővel rendelkeznek alapból, amit nekünk reprodukálni kell az AI-ba. Ezek sokszor egyszerűsített modellek is elegek, mint pl. a számítógépes játékok fizikai szimulációs motorjai.


Overcoming Catastrophic Forgetting with Hard Attention to the Task
http://proceedings.mlr.press/v80/serra18a.html

A probléma: hogy amikor megtanulunk egy feladatot, a régebbiek átsúlyozódnak, és kevésbé hatékonyan előhívhatóak lesznek. Megoldási javaslat: úgy kell tanítnai, hogy az adott feladat tanulása során nem minden súlyt módosítunk. Az eljárás megnézi, hogy mik voltak a fontos dolgok (súlyok) a korábib tanításokban és utána a következő tanításban azokat engedi módosítani, akiket eddig kevésbé hasnzáltunk. Mellékhatás: ha másképp állítjuk be, akkor tudja a hálót tömöríteni is. Mindehhez nagyon kevés új súlyt kella hálóhoz tenni, és csupán 2 paramétert kell beállítani a használatához.


Poszterek: szimmetria:
Csoport invariáns konvolúciós hálózatok alapötlete:
Group Equivariant Convolutional Networks
https://arxiv.org/abs/1602.07576


Symmetry Regularization
https://dspace.mit.edu/handle/1721.1/109391

Forgás invariancia:
Robustness of Rotation-Equivariant Networks to Adversarial Perturbations
https://arxiv.org/abs/1802.06627

Sample Efficient Semantic Segmentation using Rotation Equivariant Convolutional Networks
https://arxiv.org/abs/1807.00583

Universal approximations of invariant maps by neural networks
https://arxiv.org/abs/1804.10306

Application:
3D G-CNNs for Pulmonary Nodule Detection
https://arxiv.org/abs/1804.04656

Scale invariance:

Theory:

Locally Scale-Invariant Convolutional Neural Networks
https://arxiv.org/abs/1412.5104

Learning Invariant Representations with Local Transformations
https://arxiv.org/abs/1206.6418


Rotation equivariant vector field networks
https://arxiv.org/abs/1612.09346

Application:
Land cover mapping at very high resolution with rotation equivariant CNNs: towards small yet accurate models
https://arxiv.org/abs/1803.06253


Szombat:
TherML: Thermodynamics of Machine Learning
https://arxiv.org/abs/1807.04162

Különböző ML eljárásokat egyesít egyetlen költség függvény speciális eseteibe, amelyet a termodinamika mintájára konstruál meg. Különböző eljárások más fázisokat járnak be ebben a formalizmusban, de egyiksem képes a teljes optimumot megtalálni, mert nem lépnek át a fázisokon. A random init egy nem-egyensúlyi helzyetnek felel meg, a végső optimum egyensúlyban van, ha van egy megfelleő optimizerünk, ami beállítja az egyensúlyt. Vagy, lehet mintevételezni a megfelelő Boltzmann eloszlást is... Ha ez valóban működik, akkor nagyon sok fizikai konstrukció valszeg átvihető, és azt állítják, hogy itt könnyebb is végig csinálni, dolgokat a fizikával szemben.




HUBI
zero sum competing games



nampy
example based learning>
MS: prose playground - excel

search in a dsl, prune using logical reasoning, guide using ML -> program set -> ranking -> interact with hte user to improve -> final program
dsl simple
logical reasoning> divide and conquer, function inverses
ML: which subgoal to use / improve, uses LSTM


Brenden Lake Program Induction for building more human ML systems.
concept learning, question asking.
fractal prediction

if the DSL is good enough program synthesis solves 100%

Question asking: DSL helps asking good questions and maximise information gain (bayesian learning)



Zeroshot task generalization (parametrized tasks w interruptaion, stcak etc)

Task communication, other agent executes. (Speaker - Listerner Agents)
Grammar -> Image (stacked blocks) -> other agent builds the stacks
LSTM + CNN

bandit optimization> correct at the time it makes mistake. Invariancve on ordering is not a problem with bandits, but a problem in SL, where it must be specified.



Deep Learning Perspective on Program Induction

1: NN is the program
2: NN generates source code

2015: sec2sec... for all things...
input string try to predict result of execution, not RL with a non differentiable  component, and learn the same with a network. With RL it is easier. It was almost working...

2018: Deep RL all...
1: NN generates images
2: NN generates visual programs

view change: predict new image in new viewport from old image of olda view port.
2.: infer where are the objects? Model programs instead of pixels.
libmypaint, MuJoCo
policy -> (actions) -> executed in environment -> (generate) -> image -> fed back to agent ... random images in place of all possible actions...

rewarding? GAN critic...if critic fails, reward agent...but cannot backprop (non-diff component)

inverse graphics.. reconstruct image... what loss to use? pair of target image and painted to the critic...

discriminator to push away from identity (blank).



RL + DSL do thing, fed into Deep NN for critic!
