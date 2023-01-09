# Cellaton

- [x] Rule 110
- [x] Rule 90
- [x] Rule 30
- [x] Convay game of life
- [x] Seeds
- [x] Brian's Brain
- [x] Wire World
- [x] Langton's ant
- [x] HPP Latice Gas model

Rules are done with single algorithm.

Convay, seeds, Brian's Brain and Wire World are also done with single algorithm.

Wire World is done by writing pixels to texture directly.

HPP Latice Gas Model is done with custom kernel that encodes lattice into single texture and uses "convolution-like" algorithm to determine cell behavior.

----------------------------------------------

# Links
https://github.com/Chakazul/Lenia app from some youtube video about smooth game of life

https://www.youtube.com/watch?v=8wDSQxmAyTw&ab_channel=NeatAI video from smooth gol from NeatAI (there is also link to shader toys)

https://github.com/duckythescientist/SmoothLife 

https://arxiv.org/pdf/1111.1567.pdf

https://conwaylife.com/wiki/List_of_Life-like_cellular_automata

http://www.radicaleye.com/lifepage/picgloss/picgloss.html list of Convay's gol patterns

# TODO
- [x] wire world https://en.wikipedia.org/wiki/Wireworld
- [x] gasatomato://en.wikipedia.org/wiki/Lattice_gas_automaton 
- [x] ant https://en.wikipedia.org/wiki/Langton%27s_ant
- [ ] LBM https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods
- [ ] smooth gol https://www.youtube.com/watch?v=8wDSQxmAyTw&ab_channel=NeatAI
- [ ] 3d cellular automaton https://www.youtube.com/watch?v=dQJ5aEsP6Fs&ab_channel=Softology
- [ ] Ulam's two step https://dmackinnon1.github.io/svgpixel/ulam.html

# lattice gas automaton
use rgba texture

component describe cell direction

to which component should cell write its direction if its moved

* 0 *
3 c 1
* 2 *

directions:
0 - top
1 - right
2 - bot
3 - left

also write only to current invocation pos (to remove simaltenous writes)

cell is live if:
top cell have direction bot (2) or
right cell have direction left (3) or
bot cell have direction top (0) or
left cell have direction right (1)

otherwise dead
