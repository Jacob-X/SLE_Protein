# 加载必要的库
library(forestplot)
library(dplyr)
library(readr)
library(grid)  # 确保 gpar() 可用

# 读取数据
data <- read_csv("/Users/xutianxin/codes/Proteomic_data_explore/pythonProject/10.MR/sph_sign_mr_result.csv")

# 按 BETA 值从高到低排序
data <- data %>% arrange(desc(BETA))

# 筛选出 Analysis 为 "genome-wide-IV.2way" 的数据
data <- data %>% filter(Analysis %in% c("genome-wide-IV.2way", "genome-wide-IV.1way"))
# **计算 95% 置信区间**
data$lower <- data$BETA - 1.96 * data$SE
data$upper <- data$BETA + 1.96 * data$SE

# 计算 OR 和 95% 置信区间
data$OR <- exp(data$BETA)  # 计算 OR（比值比）
data$OR_lower <- exp(data$lower)  # 计算 OR 下限
data$OR_upper <- exp(data$upper)  # 计算 OR 上限

# **P 值格式化为 10⁻ⁿ 形式**
data$P.ivw.formatted <- formatC(data$P.ivw, format = "e", digits = 2)
data$P.ivw.formatted <- gsub("e", " × 10^", data$P.ivw.formatted)  # 替换 e 为 10^

tabletext <- cbind(
  c(NA, "Exposure-Outcome", paste(data$X, "-", data$Y)),  # 第一列：暴露-结局变量
  c(NA, "OR (95% CI)", sprintf("%.3f [%.3f, %.3f]", data$OR, data$OR_lower, data$OR_upper)),  # 第二列
  c(NA, "P-value", data$P.ivw.formatted)  # 第三列
)

# **修正 mean, lower, upper 的行数**
num_rows <- nrow(tabletext)  # 获取 tabletext 的行数
mean_values <- c(NA, NA, data$OR)  # 添加 NA 以匹配行数
lower_values <- c(NA, NA, data$OR_lower)
upper_values <- c(NA, NA, data$OR_upper)
# **修正 is.summary 的行数**
is_summary <- c(TRUE, TRUE, rep(FALSE, nrow(data)))


forestplot(
  labeltext = tabletext, 
  graph.pos = 3,  # 让图放在第 3 列
  mean = mean_values, 
  lower = lower_values, 
  upper = upper_values,
  zero = 1,  # 添加参考线（OR = 1）
  boxsize = 0.3,  # 设定点大小
  xlog = TRUE,  # 让 OR 采用对数刻度，使得误差条更清晰
  col = fpColors(box = "royalblue", lines = "darkblue", zero = "gray50"),
  xlab = "OR (95% CI)",  # x 轴标签
  is.summary = is_summary,
  txt_gp = fpTxtGp(
    label = gpar(cex = 1.5),  # **放大 tabletext 里的文本**
    ticks = gpar(cex = 1.5),  # 调大 x 轴刻度字体
    xlab = gpar(cex = 1.5),   # 调大 x 轴标题字体
    title = gpar(cex = 1.5)   # 如果有标题，也调大
  ),
  align = c("c", "c", "c"),  # **让 tabletext 的所有列居中对齐**
  new_page = TRUE,  # 让森林图绘制在新页面，防止空间不足
  colgap = unit(12, "mm"),  # **增加 tabletext 和图形之间的间距**
  clip = c(0.1, 10)  # **设定 OR 轴的范围，避免置信区间过长被截断**
)