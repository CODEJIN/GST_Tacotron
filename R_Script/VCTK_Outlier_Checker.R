library(readr)
library(ggplot2)
library(car)

repeat_Count <- 1000
base_Dir = 'D:/Python_Programming/GST_Tacotron/'

vctk_Length.Data <- read_delim(
  sprintf('%sVCTK_Length.txt', base_Dir),
  "\t",
  escape_double = FALSE,
  locale = locale(encoding = "UTF-8"),
  trim_ws = TRUE
  )

vctk_Length.Sig <- vctk_Length.Data[c(-4,-5)]
vctk_Length.Trim <- vctk_Length.Data[c(-3,-5)]
vctk_Length.Split <- vctk_Length.Data[c(-3,-4)]

vctk_Length.Sig.Plot <- ggplot(vctk_Length.Sig, aes(x= Sig_Length, y= Text_Length)) +
  geom_point() +
  labs(title=sprintf('Original Sig count: %s', nrow(vctk_Length.Sig))) +
  geom_smooth(method = "lm")
vctk_Length.Trim.Plot <- ggplot(vctk_Length.Trim, aes(x= Trim_Length, y= Text_Length)) +
  geom_point() +
  labs(title=sprintf('Original Trim count: %s', nrow(vctk_Length.Trim))) +
  geom_smooth(method = "lm")
vctk_Length.Split.Plot <- ggplot(vctk_Length.Split, aes(x= Split_Length, y= Text_Length)) +
  geom_point() +
  labs(title=sprintf('Original Split count: %s', nrow(vctk_Length.Split))) +
  geom_smooth(method = "lm")

for (index in seq(repeat_Count))
{
  vctk_Length.Sig$Num <- row.names(vctk_Length.Sig)
  vctk_Length.Trim$Num <- row.names(vctk_Length.Trim)
  vctk_Length.Split$Num <- row.names(vctk_Length.Split)
  
  vctk_Length.Sig.LM <- lm(
    Sig_Length ~ Text_Length + I(Text_Length^2),
    data=vctk_Length.Sig
  )
  vctk_Length.Trim.LM <- lm(
    Trim_Length ~ Text_Length + I(Text_Length^2),
    data=vctk_Length.Trim
  )
  vctk_Length.Split.LM <- lm(
    Split_Length ~ Text_Length + I(Text_Length^2),
    data=vctk_Length.Split
  )
  
  vctk_Length.Sig.Outlier <- outlierTest(vctk_Length.Sig.LM)
  vctk_Length.Trim.Outlier <- outlierTest(vctk_Length.Trim.LM)
  vctk_Length.Split.Outlier <- outlierTest(vctk_Length.Split.LM)
  
  vctk_Length.Sig$Outlier <- vctk_Length.Sig$Num %in% as.numeric(names(vctk_Length.Sig.Outlier$p))
  vctk_Length.Trim$Outlier <- vctk_Length.Trim$Num %in% as.numeric(names(vctk_Length.Trim.Outlier$p))
  vctk_Length.Split$Outlier <- vctk_Length.Split$Num %in% as.numeric(names(vctk_Length.Split.Outlier$p))
  
  vctk_Length.Sig <- subset(vctk_Length.Sig, !Outlier)
  vctk_Length.Trim <- subset(vctk_Length.Trim, !Outlier)
  vctk_Length.Split <- subset(vctk_Length.Split, !Outlier)
}

vctk_Length.Sig.Plot.Remove_Outlier <- ggplot(vctk_Length.Sig, aes(x= Sig_Length, y= Text_Length)) +
  geom_point() +
  labs(title=sprintf('Outlier removed Sig count: %s', nrow(vctk_Length.Sig))) +
  geom_smooth(method = "lm")
vctk_Length.Trim.Plot.Remove_Outlier <- ggplot(vctk_Length.Trim, aes(x= Trim_Length, y= Text_Length)) +
  geom_point() +
  labs(title=sprintf('Outlier removed Trim count: %s', nrow(vctk_Length.Trim))) +
  geom_smooth(method = "lm")
vctk_Length.Split.Plot.Remove_Outlier <- ggplot(vctk_Length.Split, aes(x= Split_Length, y= Text_Length)) +
  geom_point() +
  labs(title=sprintf('Outlier removed Split count: %s', nrow(vctk_Length.Split))) +
  geom_smooth(method = "lm")



ggsave(
  filename = sprintf('%sSig.Original.png', base_Dir),
  plot = vctk_Length.Sig.Plot,
  device = "png", width = 12, height = 12, units = "cm", dpi = 300
)
ggsave(
  filename = sprintf('%sTrim.Original.png', base_Dir),
  plot = vctk_Length.Trim.Plot,
  device = "png", width = 12, height = 12, units = "cm", dpi = 300
)
ggsave(
  filename = sprintf('%sSplit.Original.png', base_Dir),
  plot = vctk_Length.Split.Plot,
  device = "png", width = 12, height = 12, units = "cm", dpi = 300
)
ggsave(
  filename = sprintf('%sSig.RemoveOutlier.png', base_Dir),
  plot = vctk_Length.Sig.Plot.Remove_Outlier,
  device = "png", width = 12, height = 12, units = "cm", dpi = 300
)
ggsave(
  filename = sprintf('%sTrim.RemoveOutlier.png', base_Dir),
  plot = vctk_Length.Trim.Plot.Remove_Outlier,
  device = "png", width = 12, height = 12, units = "cm", dpi = 300
)
ggsave(
  filename = sprintf('%sSplit.RemoveOutlier.png', base_Dir),
  plot = vctk_Length.Split.Plot.Remove_Outlier,
  device = "png", width = 12, height = 12, units = "cm", dpi = 300
)

write.table(vctk_Length.Trim[c(1)], sprintf('%svctk_nonoutlier.txt', base_Dir),sep='\t', row.names=FALSE, quote= FALSE)
