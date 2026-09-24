import Link from "next/link";
import {
  ArrowRight,
  BrainCircuit,
  ChartNoAxesCombined,
  ChevronRight,
  Database,
  Dna,
  FlaskConical,
  Network,
  ShieldCheck,
} from "lucide-react";
import styles from "./page.module.css";

export default function Home() {
  return (
    <main className={styles.page}>
      <nav className={styles.nav} aria-label="Main navigation">
        <Link href="/" className={styles.brand}>
          <span className={styles.brandMark}><Dna size={20} strokeWidth={2.5} /></span>
          <span>Gene<span>Web</span></span>
        </Link>
        <div className={styles.navLinks}>
          <a href="#workflow">Workflow</a>
          <a href="#capabilities">Capabilities</a>
          <Link href="/login">Sign in <ArrowRight size={15} /></Link>
        </div>
      </nav>

      <section className={styles.hero}>
        <div className={styles.heroCopy}>
          <div className={styles.eyebrow}><span /> COMPUTATIONAL BIOLOGY WORKSPACE</div>
          <h1>From gene expression to <em>actionable</em> discovery.</h1>
          <p className={styles.heroLead}>
            A unified research environment for target identification, drug-target
            analysis, and AI-assisted affinity prediction.
          </p>
          <div className={styles.heroActions}>
            <Link href="/login?mode=signup" className={styles.primaryButton}>
              Create your workspace <ArrowRight size={17} />
            </Link>
            <Link href="/login" className={styles.textButton}>
              Sign in <ChevronRight size={17} />
            </Link>
          </div>
          <div className={styles.trustLine}><ShieldCheck size={16} /> Private, authenticated research workflows</div>
        </div>
        <div className={styles.heroVisual} aria-label="Platform workflow preview">
          <div className={styles.visualGlow} />
          <div className={styles.dataOrb}><Dna size={82} strokeWidth={1} /></div>
          <div className={`${styles.signalCard} ${styles.signalTop}`}><span className={styles.signalDot} /> TARGET IDENTIFICATION <strong>ACTIVE</strong></div>
          <div className={`${styles.signalCard} ${styles.signalBottom}`}><ChartNoAxesCombined size={18} /><span>Expression signal</span><strong>0.85</strong></div>
          <div className={styles.visualGrid} />
        </div>
      </section>

      <section id="workflow" className={styles.workflowSection}>
        <div className={styles.sectionHeading}>
          <span className={styles.sectionKicker}>ONE CONNECTED PIPELINE</span>
          <h2>Research moves in one direction.</h2>
          <p>Turn complex biological data into a clear path for investigation, without stitching together disconnected tools.</p>
        </div>
        <div className={styles.workflowGrid}>
          <WorkflowCard number="01" icon={<Database />} title="Identify targets" text="Upload gene-expression data, configure preprocessing, and train models through a guided interface." />
          <WorkflowCard number="02" icon={<Network />} title="Explore interactions" text="Connect expressed genes with drug associations and molecular representations from integrated datasets." />
          <WorkflowCard number="03" icon={<FlaskConical />} title="Prioritize candidates" text="Generate embeddings and predict drug-gene affinity to focus the next stage of discovery." />
        </div>
      </section>

      <section id="capabilities" className={styles.capabilityBand}>
        <div><BrainCircuit size={22} /><span>Configurable ML experiments</span></div>
        <div><Network size={22} /><span>DepMap-informed associations</span></div>
        <div><Dna size={22} /><span>Multi-modal embeddings</span></div>
        <div><ChartNoAxesCombined size={22} /><span>Reproducible artifacts</span></div>
      </section>

      <footer className={styles.footer}>
        <span>GeneWeb</span><span>Built for computational biology research</span>
      </footer>
    </main>
  );
}

function WorkflowCard({ number, icon, title, text }: { number: string; icon: React.ReactNode; title: string; text: string }) {
  return (
    <article className={styles.workflowCard}>
      <div className={styles.cardTop}><span>{number}</span><div className={styles.cardIcon}>{icon}</div></div>
      <h3>{title}</h3>
      <p>{text}</p>
      <ChevronRight className={styles.cardArrow} size={18} />
    </article>
  );
}
