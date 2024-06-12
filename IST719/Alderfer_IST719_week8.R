link.data = read.csv(paste0(my.dir, "LINKS-421-719Network.csv")
                     , header = TRUE
                     , stringsAsFactors = F)
node.data = read.csv(paste0(my.dir, "NODES-421-719Network.csv")
                     , header = TRUE
                     , stringsAsFactors = F)

colnames(link.data)
colnames(link.data) = gsub("\\.", "", colnames(link.data))
link.data$X = gsub(" |-", "", link.data$X)
cbind(link.data$X, colnames(link.data)[-1])

node.data$Name = gsub(" |-", "", node.data$Name)
cbind(node.data$Name, link.data$X)


M = as.matrix(link.data[ , -1])
M
rownames(M) = colnames(M)
dim(M)
any(is.na(M))
M[is.na(M)] = 0
M[is.na(M)]

install.packages("igraph")
library(igraph)
g = graph_from_adjacency_matrix(M)

vcount(g)
ecount(g)

plot.igraph(g)

g = simplify(g)
par(mar=c(0,0,0,0))
plot.igraph(g, edge.arrow.size = 0, edge.arrow.width = 0)

E(g)$arrow.size = 0
E(g)$arrow.width = 0

plot.igraph(g)

V(g)$color = "gold"
V(g)$frame.color = "white"
V(g)$label.color = "black"
E(g)$color = "cadetblue"
V(g)$size = 5

plot.igraph(g)

E(g)$curved = .4

degree(g)

par(mar = c(3,10,1,1))
barplot(degree(g), horiz = T, las = 2)

V(g)$degree = degree(g)

V(g)$deg.out = degree(g, mode = "out")
V(g)$deg.in = degree(g, mode = "in")

barplot(V(g)$deg.out, names.arg = V(g)$name,
        horiz = T, las = 2)
barplot(V(g)$deg.in, names.arg = V(g)$name,
        horiz = T, las = 2)

g.bak = g
g = as.undirected(g)
g = g.bak

V(g)$close = closeness(g, normalized = T, mode = "all")
V(g)$bet = betweenness(g, directed = FALSE)

plot.igraph(g)

my.pallet = colorRampPalette(c("steelblue", "violet", "tomato", "red", "red"))

V(g)$color = rev(
  my.pallet(200))[round(1 + rescale(V(g)$close, c(1,199)), 0)]

V(g)$size = 2 + rescale(V(g)$degree, c(0,13))
V(g)$label.cex = .7 + rescale(V(g)$bet, c(0,1.25))


cbind(V(g)$name, node.data$Name)

V(g)$class = node.data$Class
V(g)$country = node.data$Country
V(g)$year = node.data$year

g = delete_vertices(g, "JoHunter")
plot.igraph(g)

V(g)$shape = "circle"
V(g)$shape[V(g)$class == "Wednesday"] = "square"
V(g)$shape[V(g)$class == "Both"] = "rectangle"


V(g)$color = "gold"
V(g)$color[V(g)$country == "India"] = "springgreen4"
V(g)$color[V(g)$country == "China"] = "red"
V(g)$color[V(g)$class == "Both"] = "purple"

V(g)$label.color = "blue"
V(g)$label.color[V(g)$year == 1] = "black"

fc = cluster_fast_greedy((as.undirected(g)))
print(modularity(fc))

membership(fc)
V(g)$cluster = membership(fc)
length(fc)
sizes(fc)

par(mar = c(0,0,0,0))
plot_dendrogram(fc, platette = rainbow(7))


load(paste0(my.dir, "ist719NetworkObject.rda"))
par(mar = c(0,0,0,0))

l = layout_in_circle(g)
V(g)$x = l[,1]
V(g)$y = l[,2]
plot.igraph(g)

l = layout_with_fr(g)
E(g)$color = "gray"
E(g)[.from("BaiNing")]$color = "red"
l = layout_as_star(g, center = "BaiNing")

V(g)$x = 0
V(g)$y = 0
l = layout_with_kk(g)

coord = cbind(V(g)$x, V(g)$y)

iterations = c(500,100,20,10,5,3,2,1)
for (i in 1:length(iterations)) {
  l = layout_with_fr(g, coords = coord, dim = 2, niter = iterations[i])
  V(g)$x = l[,1]
  V(g)$y = l[,2]
  plot.igraph(g)
  mtext(paste("Layour FR:", iterations[i]), side = 3, line = 0, cex = 1.5, adj =0)
}

l = layout_with_gem(g)
l = layout_with_dh(g)
l = layout_on_grid(g)


my.linked.list = data.frame(person = V(g)$name, event = V(g)$country)
g = graph_from_data_frame(my.linked.list, directed = F)

V(g)$type = FALSE
V(g)$type[V(g)$name %in% node.data$Name] = TRUE

l = layout_as_bipartite(g, types = V(g)$type)
V(g)$x = l[ , 2]
V(g)$y = l[ , 1]

par(mar = c(0,0,0,0))
plot.igraph(g)

V(g)$size = 0
