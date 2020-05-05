# Refer: https://statkclee.github.io/model/model-tsne.html

library(tidyverse)
library(Rtsne)
library(tools)

base_Dir <- 'D:/GST.Results/Inference/GST/'

list.files(base_Dir)
for(file in list.files(base_Dir))
{
  if (toupper(file_ext(file)) != 'TXT')
  {
    next
  }
  else if (file.exists(sprintf('%s%s', base_Dir, str_replace(file, '.TXT', '.PNG'))))
  {
    next
  }
  
  gst.Data <- read_delim(
    sprintf('%s%s', base_Dir, file),
    "\t",
    escape_double = FALSE,
    locale = locale(encoding = "UTF-8"),
    trim_ws = TRUE
  )
  gst.TSNE <- Rtsne(
    gst.Data[,c(-1)],
    PCA = TRUE,
    check_duplicates = FALSE,
    dims = 2,
    max_iter = 1000,
    perplexity= 5
  )
  gst.TSNE.DF <- data.frame(
    TSNE_x = gst.TSNE$Y[, 1], 
    TSNE_y = gst.TSNE$Y[, 2], 
    Data_Tag = gst.Data$Tag
  )
  
  
  plot <- ggplot(data= gst.TSNE.DF, aes(x= TSNE_x, y= TSNE_y, color= Data_Tag)) +
    geom_point() +
    #geom_text(aes(label= Data_Tag)) +
    labs(title= 'GST t-SNE', x= '', y= '') +
    theme_bw() +
    theme(
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      # axis.text = element_blank(),
      strip.text = element_text(size = 20),
      panel.grid=element_blank(),
      legend.position = 'right',
      plot.title = element_text(hjust = 0.5)
    )
  
  
  ggsave(
    filename = sprintf('%s%s', base_Dir, str_replace(file, '.TXT', '.PNG')),
    plot = plot,
    device = "png", width = 12, height = 10, units = "cm", dpi = 300
  )
}