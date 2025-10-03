

\# Léxico Político Brasileiro \- Contexto Bolsonaro (2019-2023)

\# Este arquivo contém termos e expressões específicas do contexto político brasileiro

brazilian\_political\_lexicon:

\# Governo Bolsonaro e apoiadores

governo\_bolsonaro:

\- "presidente"

\- "capitão"

\- "mito"

\- "bolsonaro"

\- "jair"

\- "messias"

\- "patriota"

\- "conservador"

\- "direita"

\- "família tradicional"

\- "deus pátria família"

\- "brasil acima de tudo"

\# Oposição política

oposição:

\- "lula"

\- "pt"

\- "petista"

\- "esquerda"

\- "comunista"

\- "socialista"

\- "vermelho"

\- "esquerdista"

\- "bolivariano"

\- "foro de são paulo"

\- "partido dos trabalhadores"

\# Instituições

instituições:

\- "stf"

\- "supremo tribunal federal"

\- "tse"

\- "tribunal superior eleitoral"

\- "congresso"

\- "senado"

\- "câmara"

\- "deputados"

\- "senadores"

\- "ministério"

\- "governo federal"

\- "união"

\- "poderes"

\# Teorias conspiratórias

teorias\_conspiração:

\- "urna fraudada"

\- "fraude eleitoral"

\- "ditadura stf"

\- "globalismo"

\- "nova ordem mundial"

\- "deep state"

\- "estado profundo"

\- "illuminati"

\- "maçonaria"

\- "elite globalista"

\- "kit gay"

\- "mamadeira de piroca"

\- "comunismo na educação"

\- "doutrinação"

\- "marxismo cultural"

\# Negacionismo sanitário

saúde\_negacionismo:

\- "tratamento precoce"

\- "ivermectina"

\- "cloroquina"

\- "hidroxicloroquina"

\- "ditadura sanitária"

\- "lockdown"

\- "isolamento social"

\- "passaporte sanitário"

\- "vacina experimental"

\- "covid laboratório"

\- "gripezinha"

\- "histeria coletiva"

\- "plandemia"

\- "grande reset"

\# Militarismo

militarismo:

\- "forças armadas"

\- "militares"

\- "intervenção militar"

\- "quartel"

\- "exército"

\- "marinha"

\- "aeronáutica"

\- "comandos militares"

\- "generais"

\- "coronéis"

\- "oficiais"

\- "caserna"

\- "disciplina militar"

\- "ordem e progresso"

\- "ai-5"

\- "ditadura militar"

\- "regime militar"

\# Mídia e comunicação

mídia\_comunicação:

\- "fake news"

\- "mídia golpista"

\- "grande mídia"

\- "globo"

\- "rede globo"

\- "folha"

\- "estadão"

\- "veja"

\- "manipulação"

\- "censura"

\- "verdade"

\- "imprensa marrom"

\- "jornalismo"

\- "desinformação"

\# Mobilização e ação

mobilização:

\- "acordem"

\- "despertem"

\- "brasil acorda"

\- "patriotas"

\- "manifestação"

\- "protesto"

\- "passeata"

\- "ato público"

\- "caminhoneiros"

\- "produtores rurais"

\- "empresários"

\- "classe média"

\- "povo brasileiro"

\- "nação"

\# Economia e política

economia\_política:

\- "auxílio emergencial"

\- "pix"

\- "marco temporal"

\- "agronegócio"

\- "ruralistas"

\- "mst"

\- "reforma agrária"

\- "terra"

\- "propriedade privada"

\- "livre mercado"

\- "privatização"

\- "estado mínimo"

\- "neoliberalismo"

\# Religião e valores

religião\_valores:

\- "cristão"

\- "evangélico"

\- "católico"

\- "pastor"

\- "igreja"

\- "deus"

\- "jesus"

\- "família"

\- "moral"

\- "ética"

\- "valores cristãos"

\- "tradição"

\- "conservadorismo"

\- "aborto"

\- "ideologia de gênero"

\- "lgbt"

\# Eleições e democracia

eleições\_democracia:

\- "urna eletrônica"

\- "voto impresso"

\- "auditoria"

\- "transparência"

\- "democracia"

\- "eleições limpas"

\- "soberania popular"

\- "legitimidade"

\- "golpe"

\- "impeachment"

\- "cassação"

\- "mandato"

\# Indicadores emocionais

indicadores\_emocionais:

raiva:

\- "indignação"

\- "revolta"

\- "ódio"

\- "raiva"

\- "injustiça"

\- "absurdo"

\- "inadmissível"

medo:

\- "perigo"

\- "ameaça"

\- "risco"

\- "cuidado"

\- "alerta"

\- "preocupação"

\- "medo"

esperança:

\- "mudança"

\- "futuro"

\- "esperança"

\- "fé"

\- "confiança"

\- "otimismo"

\- "vitória"

urgência:

\- "agora"

\- "urgente"

\- "rápido"

\- "imediato"

\- "já"

\- "hoje"

\- "amanhã pode ser tarde"

\# Configurações de análise

analysis\_config:

min\_term\_frequency: 2

case\_sensitive: false

include\_partial\_matches: true

weight\_by\_category: true

\# Pesos por categoria para scoring

category\_weights:

governo\_bolsonaro: 1.0

oposição: 1.0

teorias\_conspiração: 1.5

saúde\_negacionismo: 1.5

militarismo: 1.3

mobilização: 1.2

indicadores\_emocionais: 0.8

def \_initialize\_patterns(self):

"""Initialize all pre-compiled regex patterns"""

\# Political keywords (Brazilian context)

political\_keywords \= \[

'bolsonaro', 'lula', 'pt', 'psl', 'pl', 'psdb', 'mdb', 

'presidente', 'eleição', 'voto', 'política', 'brasil',

'mito', 'lula livre', 'fora bolsonaro', 'brasil acima de tudo',

'temer', 'dilma', 'moro', 'dallagnol', 'glenn', 'intercept',

'stf', 'tse', 'pf', 'mpf', 'congresso', 'senado', 'câmara',

'ministro', 'deputado', 'senador', 'governo', 'federal',

'esquerda', 'direita', 'centro', 'conservador', 'progressista',

'comunista', 'socialista', 'capitalista', 'liberal',

'democracia', 'ditadura', 'golpe', 'impeachment', 'cpi',

'corrupção', 'petrolão', 'mensalão', 'lava jato', 'propina'

\]

\# Conspiracy and misinformation keywords

conspiracy\_keywords \= \[

'fake news', 'fake', 'mentira', 'conspiração', 'teoria',

'illuminati', 'nova ordem mundial', 'globalismo', 'soros',

'bill gates', 'vacina', 'chip', '5g', 'controle mental',

'mídia manipula', 'imprensa marrom', 'grande mídia',

'deep state', 'estado profundo', 'máfia', 'esquema'

\]

\# Hate speech and extremism keywords

hate\_keywords \= \[

'comunista', 'petralha', 'mortadela', 'coxinha', 'bolsominion',

'lulaminion', 'esquerdopata', 'direitista', 'fascista',

'nazista', 'racista', 'homofóbico', 'machista', 'feminazi'

\]

\# Religious and moral keywords

religious\_keywords \= \[

'deus', 'jesus', 'cristo', 'igreja', 'pastor', 'padre',

'oração', 'fé', 'religião', 'católico', 'evangélico',

'família tradicional', 'valores cristãos', 'moral',

'aborto', 'homossexual', 'lgbt', 'ideologia de gênero'

\]

def \_initialize\_patterns(self):

"""Initialize all pre-compiled regex patterns"""

\# Political keywords (Brazilian context)

political\_keywords \= \[

'bolsonaro', 'lula', 'pt', 'psl', 'pl', 'psdb', 'mdb', 

'presidente', 'eleição', 'voto', 'política', 'brasil',

'mito', 'lula livre', 'fora bolsonaro', 'brasil acima de tudo',

'temer', 'dilma', 'moro', 'dallagnol', 'glenn', 'intercept',

'stf', 'tse', 'pf', 'mpf', 'congresso', 'senado', 'câmara',

'ministro', 'deputado', 'senador', 'governo', 'federal',

'esquerda', 'direita', 'centro', 'conservador', 'progressista',

'comunista', 'socialista', 'capitalista', 'liberal',

'democracia', 'ditadura', 'golpe', 'impeachment', 'cpi',

'corrupção', 'petrolão', 'mensalão', 'lava jato', 'propina'

\]

\# Conspiracy and misinformation keywords

conspiracy\_keywords \= \[

'fake news', 'fake', 'mentira', 'conspiração', 'teoria',

'illuminati', 'nova ordem mundial', 'globalismo', 'soros',

'bill gates', 'vacina', 'chip', '5g', 'controle mental',

'mídia manipula', 'imprensa marrom', 'grande mídia',

'deep state', 'estado profundo', 'máfia', 'esquema'

\]

\# Hate speech and extremism keywords

hate\_keywords \= \[

'comunista', 'petralha', 'mortadela', 'coxinha', 'bolsominion',

'lulaminion', 'esquerdopata', 'direitista', 'fascista',

'nazista', 'racista', 'homofóbico', 'machista', 'feminazi'

\]

\# Religious and moral keywords

religious\_keywords \= \[

'deus', 'jesus', 'cristo', 'igreja', 'pastor', 'padre',

'oração', 'fé', 'religião', 'católico', 'evangélico',

'família tradicional', 'valores cristãos', 'moral',

'aborto', 'homossexual', 'lgbt', 'ideologia de gênero'

\]

{

"ditadura": \[

100,

200,

300

\],

"DITADURA": \[

100,

200,

300

\],

"regime militar": \[

101,

201,

301

\],

"REGIME MILITAR": \[

101,

201,

301

\],

"ai-5": \[

102,

202,

302

\],

"AI-5": \[

102,

202,

302

\],

"tortura": \[

103,

203,

303

\],

"TORTURA": \[

103,

203,

303

\],

"golpe militar": \[

104

\],

"GOLPE MILITAR": \[

104

\],

"ustra": \[

105

\],

"USTRA": \[

105

\],

"repressão": \[

106,

206,

306

\],

"REPRESSÃO": \[

106,

206,

306

\],

"repressao": \[

106,

206,

306

\],

"intervenção militar": \[

107,

207,

307

\],

"INTERVENÇÃO MILITAR": \[

107,

207,

307

\],

"intervencao militar": \[

107,

207,

307

\],

"revolução": \[

108,

208

\],

"REVOLUÇÃO": \[

108,

208

\],

"revolucao": \[

108,

208

\],

"nostalgismo autoritário": \[

109

\],

"NOSTALGISMO AUTORITÁRIO": \[

109

\],

"nostalgismo autoritario": \[

109

\],

"covid-19": \[

110,

210,

310

\],

"covid": \[

110,

210,

310

\],

"COVID-19": \[

110,

210,

310

\],

"covid19": \[

110,

210,

310

\],

"corona": \[

110,

210,

310

\],

"sars-cov-2": \[

110,

210,

310

\],

"coronavirus": \[

110,

210,

310

\],

"tratamento precoce": \[

111,

211

\],

"TRATAMENTO PRECOCE": \[

111,

211

\],

"lockdown": \[

112,

212,

312

\],

"LOCKDOWN": \[

112,

212,

312

\],

"máscaras": \[

113,

213,

313

\],

"MÁSCARAS": \[

113,

213,

313

\],

"mascaras": \[

113,

213,

313

\],

"cloroquina": \[

114

\],

"CLOROQUINA": \[

114

\],

"negacionismo": \[

115,

215

\],

"NEGACIONISMO": \[

115,

215

\],

"hidroxicloroquina": \[

116

\],

"HIDROXICLOROQUINA": \[

116

\],

"ivermectina": \[

117,

217

\],

"IVERMECTINA": \[

117,

217

\],

"gripezinha": \[

118

\],

"GRIPEZINHA": \[

118

\],

"cpi da covid": \[

119

\],

"CPI DA COVID": \[

119

\],

"amazônia": \[

120,

220,

320

\],

"AMAZÔNIA": \[

120,

220,

320

\],

"amazonia": \[

120,

220,

320

\],

"desmatamento": \[

121,

221,

321

\],

"DESMATAMENTO": \[

121,

221,

321

\],

"indígena": \[

122,

222,

322

\],

"indigena": \[

122,

222,

322

\],

"INDÍGENA": \[

122,

222,

322

\],

"agronegócio": \[

123,

223,

323

\],

"AGRONEGÓCIO": \[

123,

223,

323

\],

"agronegocio": \[

123,

223,

323

\],

"queimadas": \[

124,

324

\],

"QUEIMADAS": \[

124,

324

\],

"ibama": \[

125

\],

"IBAMA": \[

125

\],

"indígenas": \[

126

\],

"indigenas": \[

126

\],

"INDÍGENAS": \[

126

\],

"terras indígenas": \[

127,

227,

327

\],

"TERRAS INDÍGENAS": \[

127,

227,

327

\],

"terras indigenas": \[

127,

227,

327

\],

"garimpo ilegal": \[

128,

328

\],

"GARIMPO ILEGAL": \[

128,

328

\],

"mudanças climáticas": \[

129

\],

"MUDANÇAS CLIMÁTICAS": \[

129

\],

"mudancas climaticas": \[

129

\],

"preto": \[

130,

230,

330

\],

"PRETO": \[

130,

230,

330

\],

"negro": \[

131,

231,

331

\],

"NEGRO": \[

131,

231,

331

\],

"minoria": \[

132,

232,

332

\],

"MINORIA": \[

132,

232,

332

\],

"discriminação": \[

133,

233,

333

\],

"DISCRIMINAÇÃO": \[

133,

233,

333

\],

"discriminacao": \[

133,

233,

333

\],

"cota": \[

134,

234,

334

\],

"COTA": \[

134,

234,

334

\],

"racismo estrutural": \[

135

\],

"RACISMO ESTRUTURAL": \[

135

\],

"ações afirmativas": \[

136

\],

"acoes afirmativas": \[

136

\],

"AÇÕES AFIRMATIVAS": \[

136

\],

"movimento negro": \[

137

\],

"MOVIMENTO NEGRO": \[

137

\],

"privilégio": \[

138,

238,

338

\],

"PRIVILÉGIO": \[

138,

238,

338

\],

"privilegio": \[

138,

238,

338

\],

"direitos humanos": \[

139

\],

"DIREITOS HUMANOS": \[

139

\],

"stf": \[

140,

240,

340

\],

"STF": \[

140,

240,

340

\],

"supremo tribunal": \[

140,

240,

340

\],

"supremo tribunal federal": \[

140,

240,

340

\],

"supremo": \[

140,

240,

340,

342

\],

"ministro": \[

141,

241,

341

\],

"MINISTRO": \[

141,

241,

341

\],

"congresso": \[

142,

242

\],

"CONGRESSO": \[

142,

242

\],

"senado": \[

143,

243

\],

"SENADO": \[

143,

243

\],

"alexandre de moraes": \[

144,

244,

344

\],

"alexandre moraes": \[

144,

244,

344

\],

"alex moraes": \[

144,

244,

344

\],

"ALEXANDRE DE MORAES": \[

144,

244,

344

\],

"moraes": \[

144,

244,

344

\],

"judiciário": \[

145,

245,

343

\],

"judiciario": \[

145,

245,

343

\],

"JUDICIÁRIO": \[

145,

245,

343

\],

"barroso": \[

146,

246,

346

\],

"BARROSO": \[

146,

246,

346

\],

"toffoli": \[

147,

247,

347

\],

"TOFFOLI": \[

147,

247,

347

\],

"xandão": \[

148

\],

"xandao": \[

148

\],

"XANDÃO": \[

148

\],

"cpi": \[

149

\],

"CPI": \[

149

\],

"militar": \[

150

\],

"MILITAR": \[

150

\],

"ordem": \[

151,

251,

351

\],

"ORDEM": \[

151,

251,

351

\],

"patriota": \[

152,

292,

352

\],

"PATRIOTA": \[

152,

292,

352

\],

"golpe": \[

153,

253,

353

\],

"GOLPE": \[

153,

253,

353

\],

"general": \[

154,

254,

354

\],

"GENERAL": \[

154,

254,

354

\],

"forças armadas": \[

155,

255,

355

\],

"forcas armadas": \[

155,

255,

355

\],

"ffaa": \[

155,

255,

355

\],

"ff.aa.": \[

155,

255,

355

\],

"FORÇAS ARMADAS": \[

155,

255,

355

\],

"segurança pública": \[

156

\],

"SEGURANÇA PÚBLICA": \[

156

\],

"seguranca publica": \[

156

\],

"militarização": \[

157

\],

"MILITARIZAÇÃO": \[

157

\],

"militarizacao": \[

157

\],

"soldado": \[

158,

258,

358

\],

"SOLDADO": \[

158,

258,

358

\],

"armas": \[

159

\],

"ARMAS": \[

159

\],

"comunismo": \[

160,

260,

360

\],

"COMUNISMO": \[

160,

260,

360

\],

"venezuela": \[

161

\],

"VENEZUELA": \[

161

\],

"cuba": \[

162

\],

"CUBA": \[

162

\],

"socialismo": \[

163,

262

\],

"SOCIALISMO": \[

163,

262

\],

"esquerda": \[

164

\],

"ESQUERDA": \[

164

\],

"ideologia de gênero": \[

165,

265,

365

\],

"ideologia de genero": \[

165,

265,

365

\],

"IDEOLOGIA DE GÊNERO": \[

165,

265,

365

\],

"marxismo cultural": \[

166

\],

"MARXISMO CULTURAL": \[

166

\],

"globalismo": \[

167,

287,

367

\],

"GLOBALISMO": \[

167,

287,

367

\],

"pt": \[

168

\],

"PT": \[

168

\],

"lula": \[

169

\],

"LULA": \[

169

\],

"corrupto": \[

170

\],

"CORRUPTO": \[

170

\],

"lava jato": \[

171

\],

"LAVA JATO": \[

171

\],

"propina": \[

172

\],

"PROPINA": \[

172

\],

"petrolão": \[

173

\],

"petrolao": \[

173

\],

"PETROLÃO": \[

173

\],

"mensalão": \[

174

\],

"MENSALÃO": \[

174

\],

"mensalao": \[

174

\],

"rachadinha": \[

175

\],

"RACHADINHA": \[

175

\],

"orçamento secreto": \[

176

\],

"ORÇAMENTO SECRETO": \[

176

\],

"orcamento secreto": \[

176

\],

"ministério público": \[

177

\],

"ministerio publico": \[

177

\],

"MINISTÉRIO PÚBLICO": \[

177

\],

"delação premiada": \[

178

\],

"DELAÇÃO PREMIADA": \[

178

\],

"delacao premiada": \[

178

\],

"pf": \[

179

\],

"PF": \[

179

\],

"eua": \[

180

\],

"EUA": \[

180

\],

"trump": \[

181

\],

"TRUMP": \[

181

\],

"china": \[

182

\],

"CHINA": \[

182

\],

"diplomacia": \[

183

\],

"DIPLOMACIA": \[

183

\],

"onu": \[

184,

284

\],

"ONU": \[

184,

284

\],

"israel": \[

185

\],

"ISRAEL": \[

185

\],

"mercosul": \[

186

\],

"MERCOSUL": \[

186

\],

"isolacionismo": \[

187

\],

"ISOLACIONISMO": \[

187

\],

"soberania nacional": \[

188

\],

"SOBERANIA NACIONAL": \[

188

\],

"alinhamento": \[

189

\],

"ALINHAMENTO": \[

189

\],

"deus": \[

190

\],

"DEUS": \[

190

\],

"cristão": \[

191

\],

"cristao": \[

191

\],

"CRISTÃO": \[

191

\],

"família": \[

192

\],

"familia": \[

192

\],

"FAMÍLIA": \[

192

\],

"evangélico": \[

193

\],

"EVANGÉLICO": \[

193

\],

"evangelico": \[

193

\],

"aborto": \[

194

\],

"ABORTO": \[

194

\],

"valores tradicionais": \[

195

\],

"VALORES TRADICIONAIS": \[

195

\],

"bancada evangélica": \[

196

\],

"BANCADA EVANGÉLICA": \[

196

\],

"bancada evangelica": \[

196

\],

"escola sem partido": \[

197

\],

"ESCOLA SEM PARTIDO": \[

197

\],

"homossexualidade": \[

198

\],

"HOMOSSEXUALIDADE": \[

198

\],

"costumes": \[

199

\],

"COSTUMES": \[

199

\],

"quarentena": \[

214,

314

\],

"QUARENTENA": \[

214,

314

\],

"pfizer": \[

218

\],

"PFIZER": \[

218

\],

"índio": \[

226,

326

\],

"ÍNDIO": \[

226,

326

\],

"indio": \[

226,

326

\],

"aquecimento global": \[

228

\],

"AQUECIMENTO GLOBAL": \[

228

\],

"garimpo": \[

229

\],

"GARIMPO": \[

229

\],

"racismo ambiental": \[

239

\],

"RACISMO AMBIENTAL": \[

239

\],

"comunista": \[

261,

361

\],

"COMUNISTA": \[

261,

361

\],

"marxismo": \[

263,

363

\],

"MARXISMO": \[

263,

363

\],

"terrorista": \[

264

\],

"TERRORISTA": \[

264

\],

"agenda comunista": \[

266

\],

"AGENDA COMUNISTA": \[

266

\],

"sanções": \[

285

\],

"sancoes": \[

285

\],

"SANÇÕES": \[

285

\],

"política externa": \[

286

\],

"POLÍTICA EXTERNA": \[

286

\],

"politica externa": \[

286

\],

"greta": \[

288,

388

\],

"GRETA": \[

288,

388

\],

"comunidades internacionais": \[

289

\],

"COMUNIDADES INTERNACIONAIS": \[

289

\],

"mimimi": \[

293

\],

"MIMIMI": \[

293

\],

"liberdade": \[

295

\],

"LIBERDADE": \[

295

\],

"disciplina": \[

296

\],

"DISCIPLINA": \[

296

\],

"conservador": \[

297

\],

"CONSERVADOR": \[

297

\],

"liberdade de expressão": \[

299

\],

"LIBERDADE DE EXPRESSÃO": \[

299

\],

"liberdade de expressao": \[

299

\],

"distanciamento social": \[

317

\],

"DISTANCIAMENTO SOCIAL": \[

317

\],

"vacina": \[

318

\],

"VACINA": \[

318

\],

"ong": \[

329

\],

"ONG": \[

329

\],

"racismo": \[

335

\],

"RACISMO": \[

335

\],

"SUPREMO": \[

342

\],

"senador": \[

345

\],

"SENADOR": \[

345

\],

"inquérito das fake news": \[

348

\],

"INQUÉRITO DAS FAKE NEWS": \[

348

\],

"inquerito das fake news": \[

348

\],

"ativismo judicial": \[

349

\],

"ATIVISMO JUDICIAL": \[

349

\],

"cidadão de bem": \[

355

\],

"CIDADÃO DE BEM": \[

355

\],

"cidadao de bem": \[

355

\],

"armamentismo": \[

356

\],

"ARMAMENTISMO": \[

356

\],

"excludente de ilicitude": \[

357

\],

"EXCLUDENTE DE ILICITUDE": \[

357

\],

"politicamente correto": \[

366

\],

"POLITICAMENTE CORRETO": \[

366

\],

"doutrinação": \[

368

\],

"DOUTRINAÇÃO": \[

368

\],

"doutrinacao": \[

368

\],

"deus acima de todos": \[

392

\],

"DEUS ACIMA DE TODOS": \[

392

\]

}

{

"100": {

"word": "ditadura",

"table": 1,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"101": {

"word": "regime militar",

"table": 1,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"102": {

"word": "ai-5",

"table": 1,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"103": {

"word": "tortura",

"table": 1,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"104": {

"word": "golpe militar",

"table": 1,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"105": {

"word": "ustra",

"table": 1,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"106": {

"word": "repressão",

"table": 1,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"107": {

"word": "intervenção militar",

"table": 1,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"108": {

"word": "revolução",

"table": 1,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"109": {

"word": "nostalgismo autoritário",

"table": 1,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"110": {

"word": "covid-19",

"table": 1,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"111": {

"word": "tratamento precoce",

"table": 1,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"112": {

"word": "lockdown",

"table": 1,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"113": {

"word": "máscaras",

"table": 1,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"114": {

"word": "cloroquina",

"table": 1,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"115": {

"word": "negacionismo",

"table": 1,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"116": {

"word": "hidroxicloroquina",

"table": 1,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"117": {

"word": "ivermectina",

"table": 1,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"118": {

"word": "gripezinha",

"table": 1,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"119": {

"word": "cpi da covid",

"table": 1,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"120": {

"word": "amazônia",

"table": 1,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"121": {

"word": "desmatamento",

"table": 1,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"122": {

"word": "indígena",

"table": 1,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"123": {

"word": "agronegócio",

"table": 1,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"124": {

"word": "queimadas",

"table": 1,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"125": {

"word": "ibama",

"table": 1,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"126": {

"word": "indígenas",

"table": 1,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"127": {

"word": "terras indígenas",

"table": 1,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"128": {

"word": "garimpo ilegal",

"table": 1,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"129": {

"word": "mudanças climáticas",

"table": 1,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"130": {

"word": "preto",

"table": 1,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"131": {

"word": "negro",

"table": 1,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"132": {

"word": "minoria",

"table": 1,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"133": {

"word": "discriminação",

"table": 1,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"134": {

"word": "cota",

"table": 1,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"135": {

"word": "racismo estrutural",

"table": 1,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"136": {

"word": "ações afirmativas",

"table": 1,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"137": {

"word": "movimento negro",

"table": 1,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"138": {

"word": "privilégio",

"table": 1,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"139": {

"word": "direitos humanos",

"table": 1,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"140": {

"word": "stf",

"table": 1,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"141": {

"word": "ministro",

"table": 1,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"142": {

"word": "congresso",

"table": 1,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"143": {

"word": "senado",

"table": 1,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"144": {

"word": "alexandre de moraes",

"table": 1,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"145": {

"word": "judiciário",

"table": 1,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"146": {

"word": "barroso",

"table": 1,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"147": {

"word": "toffoli",

"table": 1,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"148": {

"word": "xandão",

"table": 1,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"149": {

"word": "cpi",

"table": 1,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"150": {

"word": "militar",

"table": 1,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"151": {

"word": "ordem",

"table": 1,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"152": {

"word": "patriota",

"table": 1,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"153": {

"word": "golpe",

"table": 1,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"154": {

"word": "general",

"table": 1,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"155": {

"word": "forças armadas",

"table": 1,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"156": {

"word": "segurança pública",

"table": 1,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"157": {

"word": "militarização",

"table": 1,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"158": {

"word": "soldado",

"table": 1,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"159": {

"word": "armas",

"table": 1,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"160": {

"word": "comunismo",

"table": 1,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"161": {

"word": "venezuela",

"table": 1,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"162": {

"word": "cuba",

"table": 1,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"163": {

"word": "socialismo",

"table": 1,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"164": {

"word": "esquerda",

"table": 1,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"165": {

"word": "ideologia de gênero",

"table": 1,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"166": {

"word": "marxismo cultural",

"table": 1,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"167": {

"word": "globalismo",

"table": 1,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"168": {

"word": "pt",

"table": 1,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"169": {

"word": "lula",

"table": 1,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"170": {

"word": "corrupto",

"table": 1,

"category": 7,

"category\_name": "Corrupção e Transparência"

},

"171": {

"word": "lava jato",

"table": 1,

"category": 7,

"category\_name": "Corrupção e Transparência"

},

"172": {

"word": "propina",

"table": 1,

"category": 7,

"category\_name": "Corrupção e Transparência"

},

"173": {

"word": "petrolão",

"table": 1,

"category": 7,

"category\_name": "Corrupção e Transparência"

},

"174": {

"word": "mensalão",

"table": 1,

"category": 7,

"category\_name": "Corrupção e Transparência"

},

"175": {

"word": "rachadinha",

"table": 1,

"category": 7,

"category\_name": "Corrupção e Transparência"

},

"176": {

"word": "orçamento secreto",

"table": 1,

"category": 7,

"category\_name": "Corrupção e Transparência"

},

"177": {

"word": "ministério público",

"table": 1,

"category": 7,

"category\_name": "Corrupção e Transparência"

},

"178": {

"word": "delação premiada",

"table": 1,

"category": 7,

"category\_name": "Corrupção e Transparência"

},

"179": {

"word": "pf",

"table": 1,

"category": 7,

"category\_name": "Corrupção e Transparência"

},

"180": {

"word": "eua",

"table": 1,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"181": {

"word": "trump",

"table": 1,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"182": {

"word": "china",

"table": 1,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"183": {

"word": "diplomacia",

"table": 1,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"184": {

"word": "onu",

"table": 1,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"185": {

"word": "israel",

"table": 1,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"186": {

"word": "mercosul",

"table": 1,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"187": {

"word": "isolacionismo",

"table": 1,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"188": {

"word": "soberania nacional",

"table": 1,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"189": {

"word": "alinhamento",

"table": 1,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"190": {

"word": "deus",

"table": 1,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"191": {

"word": "cristão",

"table": 1,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"192": {

"word": "família",

"table": 1,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"193": {

"word": "evangélico",

"table": 1,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"194": {

"word": "aborto",

"table": 1,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"195": {

"word": "valores tradicionais",

"table": 1,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"196": {

"word": "bancada evangélica",

"table": 1,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"197": {

"word": "escola sem partido",

"table": 1,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"198": {

"word": "homossexualidade",

"table": 1,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"199": {

"word": "costumes",

"table": 1,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"200": {

"word": "ditadura",

"table": 2,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"201": {

"word": "regime militar",

"table": 2,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"202": {

"word": "ai-5",

"table": 2,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"203": {

"word": "tortura",

"table": 2,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"206": {

"word": "repressão",

"table": 2,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"207": {

"word": "intervenção militar",

"table": 2,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"208": {

"word": "revolução",

"table": 2,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"210": {

"word": "covid-19",

"table": 2,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"211": {

"word": "tratamento precoce",

"table": 2,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"212": {

"word": "lockdown",

"table": 2,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"213": {

"word": "máscaras",

"table": 2,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"214": {

"word": "quarentena",

"table": 2,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"215": {

"word": "negacionismo",

"table": 2,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"217": {

"word": "ivermectina",

"table": 2,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"218": {

"word": "pfizer",

"table": 2,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"220": {

"word": "amazônia",

"table": 2,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"221": {

"word": "desmatamento",

"table": 2,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"222": {

"word": "indígena",

"table": 2,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"223": {

"word": "agronegócio",

"table": 2,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"226": {

"word": "índio",

"table": 2,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"227": {

"word": "terras indígenas",

"table": 2,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"228": {

"word": "aquecimento global",

"table": 2,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"229": {

"word": "garimpo",

"table": 2,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"230": {

"word": "preto",

"table": 2,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"231": {

"word": "negro",

"table": 2,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"232": {

"word": "minoria",

"table": 2,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"233": {

"word": "discriminação",

"table": 2,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"234": {

"word": "cota",

"table": 2,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"238": {

"word": "privilégio",

"table": 2,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"239": {

"word": "racismo ambiental",

"table": 2,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"240": {

"word": "stf",

"table": 2,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"241": {

"word": "ministro",

"table": 2,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"242": {

"word": "congresso",

"table": 2,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"243": {

"word": "senado",

"table": 2,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"244": {

"word": "alexandre de moraes",

"table": 2,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"245": {

"word": "judiciário",

"table": 2,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"246": {

"word": "barroso",

"table": 2,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"247": {

"word": "toffoli",

"table": 2,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"251": {

"word": "ordem",

"table": 2,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"253": {

"word": "golpe",

"table": 2,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"254": {

"word": "general",

"table": 2,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"255": {

"word": "forças armadas",

"table": 2,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"258": {

"word": "soldado",

"table": 2,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"260": {

"word": "comunismo",

"table": 2,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"261": {

"word": "comunista",

"table": 2,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"262": {

"word": "socialismo",

"table": 2,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"263": {

"word": "marxismo",

"table": 2,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"264": {

"word": "terrorista",

"table": 2,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"265": {

"word": "ideologia de gênero",

"table": 2,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"266": {

"word": "agenda comunista",

"table": 2,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"284": {

"word": "onu",

"table": 2,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"285": {

"word": "sanções",

"table": 2,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"286": {

"word": "política externa",

"table": 2,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"287": {

"word": "globalismo",

"table": 2,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"288": {

"word": "greta",

"table": 2,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"289": {

"word": "comunidades internacionais",

"table": 2,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"292": {

"word": "patriota",

"table": 2,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"293": {

"word": "mimimi",

"table": 2,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"295": {

"word": "liberdade",

"table": 2,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"296": {

"word": "disciplina",

"table": 2,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"297": {

"word": "conservador",

"table": 2,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"299": {

"word": "liberdade de expressão",

"table": 2,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

},

"300": {

"word": "ditadura",

"table": 3,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"301": {

"word": "regime militar",

"table": 3,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"302": {

"word": "ai-5",

"table": 3,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"303": {

"word": "tortura",

"table": 3,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"306": {

"word": "repressão",

"table": 3,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"307": {

"word": "intervenção militar",

"table": 3,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"353": {

"word": "golpe",

"table": 3,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"354": {

"word": "general",

"table": 3,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"355": {

"word": "cidadão de bem",

"table": 3,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"358": {

"word": "soldado",

"table": 3,

"category": 0,

"category\_name": "Autoritarismo e Regimes Políticos"

},

"310": {

"word": "covid-19",

"table": 3,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"312": {

"word": "lockdown",

"table": 3,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"313": {

"word": "máscaras",

"table": 3,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"314": {

"word": "quarentena",

"table": 3,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"317": {

"word": "distanciamento social",

"table": 3,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"318": {

"word": "vacina",

"table": 3,

"category": 1,

"category\_name": "Pandemia e Saúde Pública"

},

"320": {

"word": "amazônia",

"table": 3,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"321": {

"word": "desmatamento",

"table": 3,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"322": {

"word": "indígena",

"table": 3,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"323": {

"word": "agronegócio",

"table": 3,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"324": {

"word": "queimadas",

"table": 3,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"326": {

"word": "índio",

"table": 3,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"327": {

"word": "terras indígenas",

"table": 3,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"328": {

"word": "garimpo ilegal",

"table": 3,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"329": {

"word": "ong",

"table": 3,

"category": 2,

"category\_name": "Meio Ambiente e Questões Indígenas"

},

"330": {

"word": "preto",

"table": 3,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"331": {

"word": "negro",

"table": 3,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"332": {

"word": "minoria",

"table": 3,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"333": {

"word": "discriminação",

"table": 3,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"334": {

"word": "cota",

"table": 3,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"335": {

"word": "racismo",

"table": 3,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"338": {

"word": "privilégio",

"table": 3,

"category": 3,

"category\_name": "Questões Raciais e Sociais"

},

"340": {

"word": "stf",

"table": 3,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"341": {

"word": "ministro",

"table": 3,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"342": {

"word": "supremo",

"table": 3,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"343": {

"word": "judiciário",

"table": 3,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"344": {

"word": "alexandre de moraes",

"table": 3,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"345": {

"word": "senador",

"table": 3,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"346": {

"word": "barroso",

"table": 3,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"347": {

"word": "toffoli",

"table": 3,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"348": {

"word": "inquérito das fake news",

"table": 3,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"349": {

"word": "ativismo judicial",

"table": 3,

"category": 4,

"category\_name": "Instituições Democráticas e Poderes"

},

"351": {

"word": "ordem",

"table": 3,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"352": {

"word": "patriota",

"table": 3,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"356": {

"word": "armamentismo",

"table": 3,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"357": {

"word": "excludente de ilicitude",

"table": 3,

"category": 5,

"category\_name": "Militarismo e Segurança"

},

"360": {

"word": "comunismo",

"table": 3,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"361": {

"word": "comunista",

"table": 3,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"363": {

"word": "marxismo",

"table": 3,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"365": {

"word": "ideologia de gênero",

"table": 3,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"366": {

"word": "politicamente correto",

"table": 3,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"367": {

"word": "globalismo",

"table": 3,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"368": {

"word": "doutrinação",

"table": 3,

"category": 6,

"category\_name": "Ideologia e Conflito Político"

},

"388": {

"word": "greta",

"table": 3,

"category": 8,

"category\_name": "Política Externa e Relações Internacionais"

},

"392": {

"word": "deus acima de todos",

"table": 3,

"category": 9,

"category\_name": "Religião, Cultura e Conservadorismo"

}

}

 \# Brazilian Political Thematic Categories (Governo Bolsonaro 2019-2023)

POLITICAL\_KEYWORDS \= {

'cat0\_autoritarismo': \[

\# Autoritarismo e Regimes Políticos

'ditadura', 'ditaduras', 'ditatorial', 'ditador', 'ditadores',

'regime militar', 'regime autoritário', 'regimes militares',

'ai-5', 'ai5', 'ato institucional', 'atos institucionais',

'tortura', 'torturador', 'torturadores', 'torturado', 'torturada',

'ustra', 'brilhante ustra', 'carlos alberto brilhante ustra',

'repressão', 'repressivo', 'repressiva', 'repressor', 'repressores',

'intervenção militar', 'intervenção federal', 'intervencionista',

'revolução', 'revolução de 64', '1964', 'movimento de 64',

'nostalgismo', 'nostálgico', 'nostálgica', 'saudosismo',

'autoritário', 'autoritária', 'autoritarismo', 'autocracia',

'censura', 'censurar', 'censurado', 'cerceamento',

'doi-codi', 'sni', 'dops', 'porão da ditadura'

\],

'cat1\_pandemia': \[

\# Pandemia e Saúde Pública

'cloroquina', 'hidroxicloroquina', 'hcq',

'ivermectina', 'azitromicina', 'zinco', 'vitamina d',

'tratamento precoce', 'kit covid', 'kit médico',

'negacionismo', 'negacionista', 'negacionistas', 'negar',

'gripezinha', 'histeria', 'histérico', 'exagero',

'cpi da covid', 'cpi da pandemia', 'covid-19', 'coronavirus',

'vacina', 'vacinas', 'vacinação', 'vacinado', 'vacinada',

'antivacina', 'anti-vacina', 'antivaxx', 'movimento antivacina',

'lockdown', 'isolamento', 'isolamento social', 'quarentena',

'máscara', 'máscaras', 'uso de máscara', 'obrigatoriedade',

'passaporte vacinal', 'passaporte sanitário', 'certificado',

'covaxin', 'coronavac', 'astrazeneca', 'pfizer', 'janssen',

'prevent senior', 'pazuello', 'mandetta', 'teich'

\],

'cat2\_meio\_ambiente': \[

\# Meio Ambiente e Questões Indígenas

'amazônia', 'amazônico', 'amazônica', 'floresta amazônica',

'desmatamento', 'desmatar', 'desmatado', 'desmatada',

'indígena', 'indígenas', 'índio', 'índios', 'índia', 'índias',

'agronegócio', 'agro', 'agropecuária', 'agropecuário',

'queimada', 'queimadas', 'fogo', 'incêndio florestal',

'ibama', 'icmbio', 'funai', 'órgãos ambientais',

'terras indígenas', 'demarcação', 'demarcar', 'reserva',

'garimpo', 'garimpo ilegal', 'garimpeiro', 'garimpeiros',

'mudanças climáticas', 'aquecimento global', 'clima',

'sustentabilidade', 'sustentável', 'desenvolvimento sustentável',

'mineração', 'minério', 'exploração mineral',

'pantanal', 'cerrado', 'mata atlântica', 'biomas',

'ricardo salles', 'salles', 'passar a boiada'

\],

'cat3\_questoes\_sociais': \[

\# Questões Raciais e Sociais

'preto', 'pretos', 'preta', 'pretas', 'pretinho', 'pretinha',

'negro', 'negros', 'negra', 'negras', 'negritude',

'afrodescendente', 'afrodescendentes', 'afro-brasileiro',

'minoria', 'minorias', 'minoritário', 'minoritária',

'discriminação', 'discriminar', 'discriminado', 'discriminada',

'cota', 'cotas', 'cotista', 'cotistas', 'sistema de cotas',

'racismo', 'racismo estrutural', 'racista', 'racistas',

'ações afirmativas', 'política afirmativa', 'inclusão',

'movimento negro', 'consciência negra', 'zumbi',

'privilégio', 'privilégios', 'privilegiado', 'privilegiada',

'direitos humanos', 'direitos fundamentais', 'dignidade',

'desigualdade', 'desigualdades', 'desigual', 'injustiça',

'quilombola', 'quilombolas', 'quilombo', 'remanescente'

\],

'cat4\_instituicoes': \[

\# Instituições Democráticas e Poderes

'stf', 'supremo', 'supremo tribunal federal', 'suprema corte',

'ministro', 'ministros', 'ministra', 'ministras',

'congresso', 'congresso nacional', 'parlamentar', 'parlamentares',

'senado', 'senado federal', 'senador', 'senadores', 'senadora',

'câmara', 'câmara dos deputados', 'deputado', 'deputados',

'alexandre de moraes', 'moraes', 'xandão',

'judiciário', 'poder judiciário', 'justiça', 'juiz', 'juízes',

'barroso', 'luís roberto barroso', 'roberto barroso',

'toffoli', 'dias toffoli', 'gilmar mendes', 'gilmar',

'cpi', 'comissão parlamentar', 'investigação',

'tse', 'tribunal superior eleitoral', 'justiça eleitoral',

'procuradoria', 'procurador', 'ministério público', 'mpf',

'tcú', 'tribunal de contas', 'cgu', 'controladoria'

\],

'cat5\_militarismo': \[

\# Militarismo e Segurança

'militar', 'militares', 'militarismo', 'militarização',

'ordem', 'ordem pública', 'lei e ordem', 'ordenamento',

'patriota', 'patriotas', 'patriotismo', 'patriótico',

'golpe', 'golpe militar', 'golpista', 'golpistas',

'general', 'generais', 'coronel', 'coronéis', 'major',

'forças armadas', 'exército', 'marinha', 'aeronáutica',

'segurança pública', 'segurança nacional', 'defesa',

'soldado', 'soldados', 'praça', 'praças', 'oficial',

'arma', 'armas', 'armamento', 'armado', 'armados',

'cac', 'atirador', 'atiradores', 'clube de tiro',

'polícia', 'policial', 'policiais', 'pm', 'polícia militar',

'intervenção', 'interventor', 'glo', 'garantia da lei'

\],

'cat6\_ideologia': \[

\# Ideologia e Conflito Político

'comunismo', 'comunista', 'comunistas', 'comunistinha',

'venezuela', 'venezuelano', 'maduro', 'chavez', 'chavismo',

'cuba', 'cubano', 'cubanos', 'fidel', 'castrismo',

'socialismo', 'socialista', 'socialistas', 'socialistinha',

'esquerda', 'esquerdista', 'esquerdistas', 'esquerdismo',

'direita', 'direitista', 'direitistas', 'conservador',

'ideologia de gênero', 'gênero', 'identidade de gênero',

'marxismo', 'marxismo cultural', 'marxista', 'marxistas',

'globalismo', 'globalista', 'globalistas', 'nova ordem',

'pt', 'petista', 'petistas', 'petismo', 'petralha',

'lula', 'lulismo', 'lulista', 'lulopetismo',

'bolsonaro', 'bolsonarismo', 'bolsonarista', 'bolsonaristas',

'psdb', 'psol', 'pcdob', 'rede', 'novo', 'mdb'

\],

'cat7\_corrupcao': \[

\# Corrupção e Transparência

'corrupto', 'corruptos', 'corrupta', 'corrupção',

'lava jato', 'lavajato', 'operação lava jato',

'propina', 'propinas', 'suborno', 'subornar',

'petrolão', 'mensalão', 'escândalo', 'escândalos',

'rachadinha', 'rachadinhas', 'fantasma', 'funcionário fantasma',

'orçamento secreto', 'emenda', 'emendas', 'emenda pix',

'ministério público', 'mp', 'mpf', 'procurador',

'delação', 'delação premiada', 'delator', 'delatores',

'pf', 'polícia federal', 'operação', 'operações',

'moro', 'sergio moro', 'sérgio moro', 'dallagnol',

'investigação', 'investigar', 'investigado', 'denúncia',

'improbidade', 'ímprobo', 'malversação', 'peculato'

\],

'cat8\_politica\_externa': \[

\# Política Externa e Relações Internacionais

'eua', 'estados unidos', 'americano', 'americana',

'trump', 'donald trump', 'biden', 'joe biden',

'china', 'chinês', 'chinesa', 'chineses', 'xi jinping',

'diplomacia', 'diplomático', 'diplomata', 'embaixador',

'onu', 'nações unidas', 'conselho de segurança',

'israel', 'israelense', 'palestina', 'palestino',

'mercosul', 'brics', 'g20', 'g7', 'ocde',

'isolacionismo', 'isolamento', 'isolado', 'isolacionista',

'soberania', 'soberania nacional', 'soberano', 'soberana',

'alinhamento', 'alinhado', 'aliança', 'aliado', 'aliados',

'rússia', 'putin', 'ucrânia', 'guerra', 'otan',

'globalização', 'multilateralismo', 'bilateralismo',

'ernesto araújo', 'itamaraty', 'chanceler', 'embaixada'

\],

'cat9\_religiao': \[

\# Religião, Cultura e Conservadorismo

'deus', 'cristo', 'jesus', 'senhor', 'divino',

'cristão', 'cristãos', 'cristã', 'cristãs', 'cristianismo',

'família', 'famílias', 'familiar', 'família tradicional',

'evangélico', 'evangélicos', 'evangélica', 'evangélicas',

'católico', 'católicos', 'católica', 'católicas', 'igreja',

'pastor', 'pastores', 'pastora', 'bispo', 'bispos',

'aborto', 'abortar', 'abortista', 'abortistas', 'pró-vida',

'valores', 'valores tradicionais', 'valores cristãos',

'bancada evangélica', 'bancada da bíblia', 'frente parlamentar',

'escola sem partido', 'doutrinação', 'ideologia', 'ideológico',

'homossexualidade', 'homossexual', 'homossexuais', 'lgbt',

'costumes', 'bons costumes', 'moral', 'moralidade',

'silas malafaia', 'edir macedo', 'valdemiro santiago'

\]

}

\# Categorias Transversais (aplicáveis a múltiplas orientações)

TRANSVERSAL\_KEYWORDS \= {

'pandemia': \[

\# Negacionismo e tratamento precoce

'cloroquina', 'hidroxicloroquina', 'ivermectina', 'azitromicina',

'tratamento precoce', 'kit covid', 'remédio', 'remédios',

'negacionismo', 'negacionista', 'negacionistas',

'gripezinha', 'histeria', 'plandemia', 'farsa',

\# Vacina e controle

'vacina', 'vacinas', 'vacinação', 'vacinar', 'vacinado',

'antivacina', 'anti-vacina', 'antivaxx', 'vachina',

'covaxin', 'coronavac', 'astrazeneca', 'pfizer',

'passaporte vacinal', 'passaporte sanitário', 'obrigatoriedade',

\# CPI e investigações

'cpi', 'cpi da covid', 'cpi da pandemia', 'renan calheiros',

'omar aziz', 'randolfe', 'prevent senior'

\],

'corrupcao': \[

\# Escândalos específicos

'corrupto', 'corruptos', 'corrupta', 'corruptas', 'corrupção',

'propina', 'propinas', 'suborno', 'mensalão', 'mensaleiro',

'petrolão', 'lava jato', 'lavajato', 'operação lava jato',

'rachadinha', 'rachadinhas', 'fantasma', 'funcionário fantasma',

'orçamento secreto', 'emenda', 'emendas', 'emenda pix',

\# Instituições e processos

'delação', 'delação premiada', 'delator', 'delatores',

'moro', 'sergio moro', 'sérgio moro', 'dallagnol', 'deltan',

'pf', 'polícia federal', 'mpf', 'ministério público',

'stf', 'supremo', 'supremo tribunal federal',

\# Figuras políticas

'aécio', 'cunha', 'eduardo cunha', 'temer', 'michel temer',

'geddel', 'cabral', 'sérgio cabral', 'pezão'

\],

'violencia\_politica': \[

\# Ataques e ameaças

'facada', 'facadas', 'adélio', 'atentado', 'atentados',

'ameaça', 'ameaças', 'ameaçar', 'ameaçado', 'ameaçada',

'intimidação', 'intimidar', 'coação', 'coagir',

\# 8 de Janeiro

'oito de janeiro', '8 de janeiro', '08/01', 'invasão',

'vândalos', 'vandalismo', 'depredação', 'depredar',

'terrorismo', 'terrorista', 'terroristas', 'golpista', 'golpistas',

\# Manifestações e confrontos

'manifestação', 'manifestações', 'protesto', 'protestos',

'ato', 'atos', 'patriota', 'patriotas', 'acampamento',

'bloqueio', 'bloqueios', 'caminhoneiro', 'caminhoneiros',

'greve', 'greves', 'paralização', 'paralisação'

\],

'fake\_news': \[

\# Desinformação

'fake', 'fake news', 'fakenews', 'mentira', 'mentiras',

'manipulação', 'manipular', 'manipulado', 'distorção',

'narrativa', 'narrativas', 'versão', 'versões',

\# Mídia e comunicação

'globo', 'globolixo', 'rede globo', 'folha', 'folha de sp',

'estadão', 'veja', 'imprensa', 'mídia', 'jornalista',

'telegram', 'whatsapp', 'zap', 'zapzap', 'grupo',

'viralizar', 'viral', 'compartilhar', 'compartilhamento',

\# Fact-checking

'checagem', 'fact-checking', 'agência', 'lupa', 'aos fatos'

\],

'economia': \[

\# Indicadores econômicos

'economia', 'econômico', 'econômica', 'pib', 'inflação',

'deflação', 'recessão', 'crescimento', 'desenvolvimento',

'dólar', 'real', 'câmbio', 'cotação', 'bolsa',

\# Políticas econômicas

'auxílio', 'auxílio emergencial', 'auxílio brasil', 'bolsa família',

'pix', 'drex', 'real digital', 'criptomoeda', 'bitcoin',

'imposto', 'impostos', 'tributação', 'tributário', 'fiscal',

'teto de gastos', 'responsabilidade fiscal', 'déficit', 'superávit',

\# Setores econômicos

'petrobras', 'petrobrás', 'pré-sal', 'petróleo', 'gasolina',

'combustível', 'combustíveis', 'energia', 'elétrica', 'conta de luz',

'banco central', 'bacen', 'selic', 'juros', 'crédito'

\],

'religiao': \[

\# Denominações e lideranças

'evangélico', 'evangélicos', 'evangélica', 'evangélicas',

'católico', 'católicos', 'católica', 'católicas',

'pastor', 'pastores', 'pastora', 'bispo', 'bispos',

'padre', 'padres', 'papa', 'vaticano', 'igreja',

'universal', 'assembleia de deus', 'batista', 'presbiteriana',

'silas malafaia', 'edir macedo', 'valdemiro santiago',

\# Pautas religiosas

'aborto', 'abortar', 'abortista', 'pró-vida', 'pro-vida',

'ideologia de gênero', 'gênero', 'lgbt', 'gay', 'homossexual',

'família tradicional', 'valores cristãos', 'moral', 'bons costumes',

'escola sem partido', 'doutrinação', 'marxismo cultural'

\]

}

{ "0": "Autoritarismo e Regimes Políticos", "1": "Pandemia e Saúde Pública", "2": "Meio Ambiente e Questões Indígenas", "3": "Questões Raciais e Sociais", "4": "Instituições Democráticas e Poderes", "5": "Militarismo e Segurança", "6": "Ideologia e Conflito Político", "7": "Corrupção e Transparência", "8": "Política Externa e Relações Internacionais", "9": "Religião, Cultura e Conservadorismo"

POLITICAL\_KEYWORDS \= {

    'cat0\_autoritarismo\_regime': \[

        'ai-5', 'regime militar', 'ditadura', 'tortura', 'repressão', 'intervenção militar',

        'estado de sítio', 'golpe', 'censura', 'doutrina de segurança nacional'

    \],

    'cat2\_pandemia\_covid': \[

        'covid-19', 'corona', 'pandemia', 'quarentena', 'lockdown', 'tratamento precoce',

        'cloroquina', 'ivermectina', 'máscara', 'máscaras', 'oms', 'pfizer', 'vacina',

        'passaporte sanitário'

    \],

    'cat3\_violencia\_seguranca': \[

        'criminalidade', 'segurança pública', 'violência', 'bandidos', 'facções', 'polícia',

        'militarização', 'armas', 'desarmamento', 'legítima defesa'

    \],

    'cat4\_religiao\_moral': \[

        'família tradicional', 'valores cristãos', 'igreja', 'pastor', 'padre', 'bíblia',

        'cristofobia', 'marxismo cultural', 'ideologia de gênero'

    \],

    'cat6\_inimigos\_ideologicos': \[

        'comunista', 'comunismo', 'esquerdista', 'petista', 'pt', 'lula', 'stf', 'supremo',

        'globo', 'mídia lixo', 'sistema', 'globalista', 'china', 'urss', 'cuba', 'venezuela',

        'narcoditadura', 'esquerda', 'progressista'

    \],

    'cat6\_identidade\_politica': \[

        'bolsonaro', 'bolsonarista', 'direita', 'conservador', 'patriota', 'verde e amarelo',

        'mito', 'liberdade', 'intervencionista', 'cristão', 'antiglobalista'

    \],

    'cat7\_meio\_ambiente\_amazonia': \[

        'amazônia', 'reserva', 'queimadas', 'desmatamento', 'ong', 'soberania nacional',

        'clima', 'aquecimento global', 'agenda 2030'

    \]

}

TRANSVERSAL\_KEYWORDS \= {

    'emocional\_moral': \[

        'corrupção', 'liberdade', 'patriotismo', 'soberania', 'criminoso', 'traidor',

        'bandido', 'herói', 'santo', 'vítima', 'injustiça'

    \],

    'antissistema\_deslegitimacao': \[

        'sistema', 'establishment', 'corrupto', 'imprensa vendida', 'mídia lixo', 'stf ativista',

        'conspiração', 'globalista', 'ditadura do judiciário', 'deep state'

    \],

    'polarizacao\_afetiva': \[

        'nós contra eles', 'vergonha', 'ódio', 'orgulho', 'traição', 'luta do bem contra o mal',

        'defensores da pátria', 'inimigos do povo'

    \]

}

