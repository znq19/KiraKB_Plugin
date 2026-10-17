/* 临极档案 · 八个虚构组织。
   Original fiction inspired by the bureaucratic occult mood of Project Moon;
   no canon characters or scenes are reproduced. */
window.LINJI_ORGS = {
  violet: {
    theme: "violet", chapter: "01", name: "紫垣档案局", en: "VIOLET CHAPTER",
    color: "#b58cff", badge: "assets/linji/violet/badge.webp",
    walls: ["assets/linji/violet/wall-01.webp","assets/linji/violet/wall-02.webp","assets/linji/violet/wall-03.webp","assets/linji/violet/wall-04.webp","assets/linji/violet/wall-05.webp"],
    motto: "记录不是记忆，记录是记忆的尸体。",
    summary: "临极档案的守门人。负责登记、编号、封存那些不该被城市重新想起的东西。",
    lore: "紫垣档案局没有创始人姓名。十九年前的「静默事故」后，第一任档案员把自己的名字从所有登记册上划掉，只留下一个钥匙孔形状的印章。此后每一个入职者都要在无名页上签名，承认自己从此只是索引的一部分。档案局不审判、不追捕，只负责让事物留在它该在的架子上。但最近，地下七层的编号开始自己重排，像有什么东西在借阅自己。",
    doctrine: "不被记录的事物会回来；被记录的事物也会回来，只是会带着编号。",
    uniform: "长款紫色档案员大衣、高领、黑手套、银色窄腰带、黑色长袜与踝靴。",
    relations: "与 Zwei 青盾共享现场封条；拒绝向 Hana 灰庭提交第零章的空白页。",
    quote: "不要问架子上为什么少了一本书。先问，是谁把它借走了。",
    tags: ["档案", "记忆", "第零章"]
  },
  pink: {
    theme: "pink", chapter: "02", name: "绯灯救济院", en: "CRIMSON LAMP",
    color: "#ff8fcf", badge: "assets/linji/pink/badge.webp",
    walls: ["assets/linji/pink/wall-01.webp","assets/linji/pink/wall-02.webp","assets/linji/pink/wall-03.webp","assets/linji/pink/wall-04.webp","assets/linji/pink/wall-05.webp"],
    motto: "疼痛可以被麻醉，记忆不行。",
    summary: "收容记忆污染者的救济机构。她们的灯以旧记忆为燃料，照亮的却是别人的余生。",
    lore: "绯灯救济院最初只是一间收容被「静默事故」吓坏了的孩子的病房。修女们发现，把患者最痛的一段记忆点燃，灯就会亮，病人就会安静。如今救济院拥有整条玫瑰色的走廊，每一盏灯都对应一个被自愿交出的昨天。院方坚持这叫治疗，不叫删除；但走廊尽头有一扇永远锁着的门，门后堆着没有名字的灯。",
    doctrine: "仁慈不是让人忘记，而是让人有能力继续记得。",
    uniform: "玫瑰白救济院外套、束腰、绯红臂章、黑色长袜与系带长靴。",
    relations: "为六协会处理战后创伤；与七协会共享病例索引；紫垣拒绝接收她们的灯。",
    quote: "如果一段记忆只会让你在夜里醒来，你愿意用它换一盏灯吗？",
    tags: ["救济", "记忆污染", "灯"]
  },
  blue: {
    theme: "blue", chapter: "03", name: "Zwei 青盾分局", en: "ZWEI BLUE SHIELD",
    color: "#66c7ff", badge: "assets/linji/blue/badge.webp",
    walls: ["assets/linji/blue/wall-01.webp","assets/linji/blue/wall-02.webp","assets/linji/blue/wall-03.webp","assets/linji/blue/wall-04.webp","assets/linji/blue/wall-05.webp"],
    motto: "秩序不是没有裂缝，是裂缝有人守着。",
    summary: "临极市的治安协会，负责封控、取证与保护档案。蓝盾之下，所有事故都必须有一个编号。",
    lore: "Zwei 青盾分局的制服沿用了旧城警署的剪裁：深蓝短外套、银盾徽、勤务腰带和长靴。他们不是军队，也不是纯粹的警察；他们同时负责把「不该存在于现场的东西」装进证物箱。分局内部有一条不成文的规定：进入档案地下层之前，必须把配枪留在门外。因为有些东西一旦被开枪，就会被城市正式承认为存在。",
    doctrine: "先封锁，再命名；先保护现场，再保护自己。",
    uniform: "韩式警装剪裁的深蓝短外套、银盾徽、勤务腰带、短裙、黑色长袜与长筒靴。",
    relations: "与刘协会共同处理火灾现场；向 Hana 灰庭提交月度事故报告；对紫垣保持礼貌的不信任。",
    quote: "不要对没有编号的东西开枪。那等于替它写下了第一行档案。",
    tags: ["治安", "封控", "证物"]
  },
  amber: {
    theme: "amber", chapter: "04", name: "刘协会 · 燧火", en: "LIU EMBER",
    color: "#ffb547", badge: "assets/linji/amber/badge.webp",
    walls: ["assets/linji/amber/wall-01.webp","assets/linji/amber/wall-02.webp","assets/linji/amber/wall-03.webp","assets/linji/amber/wall-04.webp","assets/linji/amber/wall-05.webp"],
    motto: "火不分辨善恶，火只分辨还能不能烧。",
    summary: "以锻造与净化闻名的协会。他们焚烧被污染的街区，也锻造封存污染的器具。",
    lore: "刘协会的燧火来自一座从未熄灭的城市熔炉。每一把交付给其它协会的封存器具，都要在炉前经过一次「认领」：锻造者把手伸进火里，念出使用者的名字。手不会受伤，名字会。燧火相信，工具必须记得谁在用它，否则它迟早会自己选择主人。最近炉温连续十九天没有升高，炉膛里却传出了第零章的朗读声。",
    doctrine: "能锻造的东西，就能封存；能封存的东西，终有一天会想出来。",
    uniform: "琥珀色防火长大衣、金属护臂、黑色束具、黑色长袜与厚重长靴。",
    relations: "为 Zwei 制作封控器械；替六协会修补 containment 链；与紫垣争夺炉渣里的残页。",
    quote: "把手伸进来。火会告诉你，你是不是一个值得被记住的名字。",
    tags: ["锻造", "净化", "熔炉"]
  },
  sg: {
    theme: "sg", chapter: "05", name: "时计局 · 世界线观测所", en: "HOROLOGE BUREAU",
    color: "#e0a340", badge: "assets/linji/sg/badge.webp",
    walls: ["assets/linji/sg/wall-01.webp","assets/linji/sg/wall-02.webp","assets/linji/sg/wall-03.webp","assets/linji/sg/wall-04.webp","assets/linji/sg/wall-05.webp"],
    motto: "世界线不会断裂，它只会被忘记自己曾经分岔。",
    summary: "观测并修正世界线的秘密机关。琥珀色的钟面，青绿色的读数，和一间不允许迟到的观测室。",
    lore: "时计局的正式名称是「临极标准时计观测所」，但没有人在日常里这样叫它。局内保存着十九年前「静默事故」前后的全部时间残差；每一枚怀表都对应一条被修正过的分岔。局员相信，城市不是被事件推动的，而是被记录事件的方式推动的。他们与紫垣共享同一套编号，却从不承认第零章存在——因为一旦承认，所有被修正过的世界线都要重新计算。",
    doctrine: "不要问现在几点。问，你正在哪一条世界上。",
    uniform: "琥珀色计时员大衣、青绿内衬、黄铜怀表与线缆、黑色长袜与长靴。",
    relations: "向七协会购买证词时间戳；与紫垣互相审核编号；Hana 灰庭每年审计一次钟差。",
    quote: "迟到不是时间问题。是你在另一条世界上，做出了不同的选择。",
    tags: ["世界线", "时钟", "观测"]
  },
  green: {
    theme: "green", chapter: "06", name: "七协会 · 苍叶情报局", en: "SEVEN VERDANT",
    color: "#8bd46f", badge: "assets/linji/green/badge.webp",
    walls: ["assets/linji/green/wall-01.webp","assets/linji/green/wall-02.webp","assets/linji/green/wall-03.webp","assets/linji/green/wall-04.webp","assets/linji/green/wall-05.webp"],
    motto: "没有沉默的证人，只有还没被归档的证词。",
    summary: "情报、取证与审讯协会。他们在温室里保存证词，让植物替死者继续说话。",
    lore: "苍叶情报局的温室里种着一种没有学名的藤。把死者的遗物埋进土里，藤会沿着残留的记忆生长，在叶片背面留下可读的纹路。七协会称它为「证词植物」，并严禁任何人把活人埋进去。局内最古老的温室不对访客开放，里面只有一株藤和一张空椅子。藤的根部缠着一枚紫垣的旧徽章，徽章上的编号被人为磨掉了。",
    doctrine: "证据不会说话；但证据会让别的东西开口。",
    uniform: "深绿情报官风衣、窄领带、皮革胸带、黑色长袜与长靴。",
    relations: "为 Zwei 提供现场分析；与绯灯交换病例；替刘协会鉴定炉渣来源。",
    quote: "植物不会撒谎。它们只会长成有人希望它们长成的样子。",
    tags: ["情报", "取证", "温室"]
  },
  red: {
    theme: "red", chapter: "07", name: "六协会 · 赤烬", en: "SIX CINDER",
    color: "#ff6b78", badge: "assets/linji/red/badge.webp",
    walls: ["assets/linji/red/wall-01.webp","assets/linji/red/wall-02.webp","assets/linji/red/wall-03.webp","assets/linji/red/wall-04.webp","assets/linji/red/wall-05.webp"],
    motto: "有些东西不该被理解，只该被关回去。",
    summary: "收容与战斗协会。他们处理「未记录者」，也是唯一被允许在档案局地下层携带武器的协会。",
    lore: "六协会的赤烬制服上永远有洗不掉的烟味。他们的工作不是杀死异常，而是把异常重新推回「未被记录」的状态；因此每一次收容结束后，队员必须亲手烧掉自己的行动记录，只留下一个数字。数字由 Hana 灰庭保管，连紫垣也不能查阅。最近一次收容失败后，赤烬的队长开始在自己的手臂上记下那些被烧掉的数字。",
    doctrine: "如果它没有名字，就不要给它一个。",
    uniform: "绯红收容战斗大衣、重型束具、锁链饰件、黑色长袜与加固长靴。",
    relations: "受 Hana 灰庭直接调遣；与 Zwei 共同封控；对紫垣隐瞒收容现场；向刘协会订购封存链。",
    quote: "我们不是去杀它。我们是去让它重新变成一个没人记得的传闻。",
    tags: ["收容", "未记录者", "赤烬"]
  },
  gray: {
    theme: "gray", chapter: "08", name: "Hana 灰庭", en: "HANA GRAY COURT",
    color: "#c7c6bd", badge: "assets/linji/gray/badge.webp",
    walls: ["assets/linji/gray/wall-01.webp","assets/linji/gray/wall-02.webp","assets/linji/gray/wall-03.webp","assets/linji/gray/wall-04.webp","assets/linji/gray/wall-05.webp"],
    motto: "契约不是承诺，契约是双方都同意忘记的例外。",
    summary: "协会之上的仲裁庭。负责执行「临极协定」，也负责决定哪些事故不配拥有档案。",
    lore: "Hana 灰庭没有常设的审判席，只有一张长桌和八把椅子。八把椅子中有一把永远空着，椅背刻着第零章。灰庭的仲裁员相信，城市之所以还能运转，不是因为所有人都遵守契约，而是因为所有人都相信别人会遵守。他们保管着赤烬的行动数字、时计局的钟差、紫垣的空白页，却从不把它们放在同一个柜子里。",
    doctrine: "不要问契约是否公平。问，谁有能力让它显得公平。",
    uniform: "灰色仲裁员套装、长外套、银色天平徽、黑色长袜与高跟长靴。",
    relations: "监督其余七个协会；向紫垣索取第零章；向时计局购买世界线审计；向六协会下达收容许可。",
    quote: "我们不需要真相。我们需要一个所有人都能接受的版本。",
    tags: ["仲裁", "契约", "临极协定"]
  }
};
